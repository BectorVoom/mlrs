---
phase: 09-spectral-family
plan: 02
type: execute
wave: 1
depends_on: ["09-01"]
files_modified:
  - crates/mlrs-kernels/src/elementwise.rs
  - crates/mlrs-backend/src/prims/laplacian.rs
  - crates/mlrs-backend/tests/laplacian_test.rs
autonomous: true
requirements: [PRIM-09]
must_haves:
  truths:
    - "laplacian(pool, A, n) returns (L, dd) where L = I − D^-1/2 A D^-1/2 matches a host reference for f32 and f64"
    - "A zero-degree (isolated) node produces NO NaN and NO inf — its dd guard is 1, its L row is 0, its L diagonal is 0"
    - "The diagonal of A is zeroed BEFORE the degree row-sum (scipy _laplacian_dense order)"
    - "The degree vector is a single-owner GATHER row reduction (no scatter, no atomics)"
    - "The dense n×n Laplacian stays in GLOBAL memory — the laplacian_map kernel is SharedMemory-free and uses no infinity constant"
    - "A PoolStats memory gate proves bounded reuse with no mid-pipeline metered readback"
  artifacts:
    - path: "crates/mlrs-kernels/src/elementwise.rs"
      provides: "laplacian_map device kernel: out[i,j] = -a/(dd_i*dd_j) off-diag, diagonal = 1-isolated"
      contains: "laplacian_map"
    - path: "crates/mlrs-backend/src/prims/laplacian.rs"
      provides: "host orchestration: zero-diag → row_reduce(Sum) degree → typed-zero dd guard → laplacian_map"
      contains: "row_reduce"
    - path: "crates/mlrs-backend/tests/laplacian_test.rs"
      provides: "value + zero_degree + memory_gate tests (un-ignored)"
      contains: "memory_gate"
  key_links:
    - from: "crates/mlrs-backend/src/prims/laplacian.rs"
      to: "crate::prims::reduce::row_reduce"
      via: "degree = row_reduce(A, Sum)"
      pattern: "row_reduce"
    - from: "crates/mlrs-backend/src/prims/laplacian.rs"
      to: "mlrs_kernels::laplacian_map"
      via: "L build map"
      pattern: "laplacian_map"
---

<objective>
PRIM-09: implement the normalized graph-Laplacian primitive and its one new
SharedMemory-free device map kernel, then standalone-validate it BEFORE any
estimator consumes it (the primitive-first gate, mirroring Phase 7/8).

`laplacian.rs` is a thin host orchestration over already-validated prims (the
kernel_matrix.rs base-op→in-place-map idiom): it RECEIVES a ready affinity `A`
(n×n) and RETURNS `(L, dd)`, reproducing scipy `_laplacian_dense` exactly:
zero the diagonal → degree row-sum via `row_reduce(Sum)` (GATHER) → typed-zero
guard `dd = where(w==0, 1, sqrt(w))` (NO F::INFINITY) → `L = I − D^-1/2 A D^-1/2`
with the isolated-node diagonal forced to 0.

Purpose: a validated, no-NaN/inf, memory-gated Laplacian that Waves 2/3 wire into.
Output: laplacian_map kernel, laplacian.rs prim, three green laplacian_test cases.
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
@crates/mlrs-backend/src/prims/kernel_matrix.rs
@crates/mlrs-kernels/src/elementwise.rs
@crates/mlrs-backend/src/prims/reduce.rs
@crates/mlrs-backend/tests/memory_gate_test.rs
@crates/mlrs-backend/tests/kernel_matrix_test.rs

# CubeCL manual MUST be read before writing the kernel (AGENTS.md §3):
# /home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md
# On ANY build error: /home/user/Documents/workspace/cintx/docs/cubecl_error_guideline.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: laplacian_map device kernel (SharedMemory-free) + the dd typed-zero guard</name>
  <read_first>
    - crates/mlrs-kernels/src/elementwise.rs: rbf_map (:114) for the #[cube(launch)] shape; div_by_row (:301) for the per-row gather divisor; clamp_nonneg (:45) / kde_epanechnikov_map (:194) for STATEMENT-form guards
    - /home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md (read BEFORE writing the kernel — AGENTS.md §3)
  </read_first>
  <behavior>
    - laplacian_map: for tid = i*n+j with i!=j, output = -a[i*n+j] / (dd[i] * dd[j]); for i==j, output = if degree_i==0 {0} else {1} (= 1 - isolated). Implemented as STATEMENT-form (the isolated case is detected via dd[i]==1 AND the row being all-zero, OR by passing the isolated mask — choose the gather-clean form; dd[i] is the guard vector so dd is never 0).
    - dd guard (separate tiny map or host step): dd[i] = if w[i]==0 {1} else {sqrt(w[i])}. STATEMENT-form, NEVER F::INFINITY, NEVER 1/sqrt(0).
    - The kernel is SharedMemory-free, atomics-free, bounds-checked (tid < a.len()), no infinity constant. n passed as a scalar u32 by value (cubecl 0.10, no ScalarArg wrapper per [02-03]).
  </behavior>
  <action>
    Read the CubeCL manual first (AGENTS.md §3). Implement `laplacian_map` in
    elementwise.rs replacing the Wave-0 stub body. Follow the rbf_map #[cube(launch)]
    signature and the div_by_row gather-divisor idiom. The off-diagonal writes
    `-a[i*n+j] / (dd[i]*dd[j])`; the diagonal writes `1 - isolated_i` where the
    isolated flag derives from the degree (a zero-degree node has dd guarded to 1 and
    must get diagonal 0). Use a STATEMENT-form guard like clamp_nonneg/kde_epanechnikov_map
    — NEVER F::INFINITY, NEVER divide by a raw degree (always the guarded dd).

    Also implement the `dd = where(w==0, 1, sqrt(w))` step. Prefer a tiny device guard
    kernel (mirror sqrt_elem:61 + a statement guard) so the whole pipeline stays
    device-resident; if a device select is awkward in cubecl 0.10, a host map over the
    metered-read degree vector is acceptable (it is length-n only). Document the choice.

    Re-export laplacian_map (and any new guard kernel) via `pub use` in
    mlrs-kernels/src/lib.rs. Doc comments must avoid the literal grep-gate tokens
    (SharedMemory / F::INFINITY) per [08-02] Rule 3 — the constructs are genuinely absent.

    On ANY build error, consult /home/user/Documents/workspace/cintx/docs/cubecl_error_guideline.md
    before attempting a fix.
  </action>
  <verify>
    <automated>cargo build --features cpu -p mlrs-kernels 2>&1 | tail -3 && grep -q "laplacian_map" crates/mlrs-kernels/src/lib.rs && ! grep -n "INFINITY" crates/mlrs-kernels/src/elementwise.rs && echo KERNEL_OK</automated>
  </verify>
  <acceptance_criteria>
    - mlrs-kernels builds under --features cpu.
    - laplacian_map and the dd guard are re-exported; no `INFINITY` token in elementwise.rs.
    - Kernel is SharedMemory-free and atomics-free (manual inspection + grep-clean).
  </acceptance_criteria>
  <done>laplacian_map compiles, is re-exported, SharedMemory-free, infinity-free; the dd typed-zero guard exists.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: laplacian.rs host orchestration + value/zero_degree/memory_gate tests</name>
  <read_first>
    - crates/mlrs-backend/src/prims/kernel_matrix.rs (:42-52 imports, :133-142 geometry guard, :196-227 launch_map_in_place + launch_dims_1d)
    - crates/mlrs-backend/src/prims/reduce.rs (:180 row_reduce, ScalarOp::Sum, ReducePath::Shared)
    - crates/mlrs-backend/tests/memory_gate_test.rs (PoolStats build-failing gate; live_bytes/peak_bytes/reuses)
    - crates/mlrs-backend/tests/kernel_matrix_test.rs (value-vs-host-reference shape, assert_slice_close, skip_f64_with_log)
    - The committed isolated-node + value laplacian .npz fixtures (Wave 0, tests/fixtures/)
  </read_first>
  <behavior>
    - laplacian(pool, A, n) -> (L, dd): step 1 zero diagonal of A (in-place map, tid where row==col → 0); step 2 w = row_reduce(A, n, n, Sum, Shared); step 3 dd = guard(w); step 4 L via laplacian_map(A, dd, out, n). Returns L (n×n) and dd (length n).
    - value test: L matches the scipy _laplacian_dense host reference within tolerance (f64 strict 1e-5; f32 documented band) on a normal fixture, f32 AND f64.
    - zero_degree test: an isolated-node fixture (a row/col of all zeros) produces L with NO NaN, NO inf; that node's dd==1, its L row all-zero, its L diagonal == 0.
    - memory_gate test: drive laplacian N=5× at fixed shape; assert bounded allocation reuse (alloc delta==0 after warmup) and no per-call mid-pipeline metered readback (mirror memory_gate_test.rs gate-2 form).
  </behavior>
  <action>
    Implement the laplacian.rs compute path (replace the Wave-0 todo!()), following the
    kernel_matrix.rs base-op→in-place-map orchestration:
    1. zero the diagonal of A FIRST (RESEARCH "Affinity diagonal handling" CONFIRMED —
       scipy fill_diagonal(m,0) before degree). An in-place map kernel or a reuse of an
       existing diag-zero idiom; the diagonal-zero must precede the degree.
    2. w = row_reduce(A, n, n, ScalarOp::Sum, ReducePath::Shared) — the degree GATHER
       (reduce.rs:180; single-owner, no scatter/atomics). NOTE: ReducePath::Shared's reduce
       kernel uses SharedMemory but is already cpu-validated — only the NEW laplacian_map
       must be SharedMemory-free.
    3. dd = guard(w) (Task-1 dd guard) — typed-zero, NO F::INFINITY.
    4. L = laplacian_map(A, dd, n) — the off-diag -a/(dd_i*dd_j) + diagonal 1-isolated.
       Per RESEARCH Anti-Pattern, you MAY thread the affinity buffer as the eig `out` later;
       here the map runs over the affinity buffer in place (launch_map_in_place idiom).
    Return (L, dd) — dd is REQUIRED by the estimators for the D-07 /dd recovery.

    Keep the geometry guard from Wave 0 (already real). The dense n×n L stays in GLOBAL
    memory — no SharedMemory tile (gfx1100 LDS ≤ 65536 B); LDS-budget is trivially satisfied
    (no tile). On ANY build error consult the cubecl_error_guideline.

    Then un-ignore and implement the three laplacian_test cases (value, zero_degree,
    memory_gate) per <behavior>. The host reference is scipy _laplacian_dense (pinned in
    RESEARCH Code Examples). f64 strict 1e-5 via skip_f64_with_log; f32 documented band
    (~1e-4 per Phase-8 precedent). The memory_gate mirrors memory_gate_test.rs.
  </action>
  <verify>
    <automated>cargo test --features cpu -p mlrs-backend laplacian_test 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test --features cpu -p mlrs-backend laplacian_test` is green (value f32+f64, zero_degree, memory_gate).
    - zero_degree fixture produces NO NaN/inf (isolated node dd==1, L row 0, diagonal 0).
    - memory_gate proves bounded reuse + no mid-pipeline metered readback.
  </acceptance_criteria>
  <done>laplacian.rs reproduces scipy _laplacian_dense (value-matched f32+f64), the zero-degree node is NaN/inf-free, and the PoolStats gate is green — the primitive-first gate is satisfied.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host orchestration → laplacian_map kernel | Buffer sizes (n) and degree values cross into the device map |
| degree value → dd guard | A zero degree must not produce 1/sqrt(0) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-9-LAP | Tampering (silent wrong result) + DoS (cpu-MLIR panic) | `1/sqrt(0)` on a zero-degree node | mitigate | Typed-zero guard `dd = where(w==0,1,sqrt(w))` + diagonal forced to 0 for isolated nodes; STATEMENT-form, NEVER `F::INFINITY` — verified by the zero_degree test (no NaN/inf) and a grep that elementwise.rs has no infinity token. |
| T-9-LDS | DoS | dense `n×n` Laplacian on a tiled SharedMemory kernel | mitigate | The new `laplacian_map` is SharedMemory-free; the dense L stays in GLOBAL memory (gfx1100 LDS ≤ 65536 B). Only the pre-existing, already-cpu-validated `ReducePath::Shared` reduce uses SharedMemory. LDS-budget audit: no new tile. |
| T-9-MEM | DoS (unbounded allocation) | repeated laplacian calls | mitigate | PoolStats memory gate (alloc delta==0 after warmup, no per-call mid-pipeline metered readback) mirroring `memory_gate_test.rs`. |
</threat_model>

<verification>
- `cargo test --features cpu -p mlrs-backend laplacian_test` green: value (f32+f64),
  zero_degree (no NaN/inf), memory_gate (bounded).
- `grep -n INFINITY crates/mlrs-kernels/src/elementwise.rs` returns nothing.
- The laplacian_map kernel contains no SharedMemory construct (manual inspection).
- laplacian returns BOTH L and dd (dd consumed by the D-07 recovery downstream).
</verification>

<success_criteria>
- PRIM-09 standalone-validated BEFORE any estimator wiring (primitive-first gate).
- L = I − D^-1/2 A D^-1/2 matches scipy _laplacian_dense within tolerance (f32+f64).
- Zero-degree nodes are NaN/inf-free via the typed-zero guard.
- The new kernel is SharedMemory-free, atomics-free, infinity-free; PoolStats gate green.
</success_criteria>

<output>
Create `.planning/phases/09-spectral-family/09-02-SUMMARY.md` when done.
</output>
