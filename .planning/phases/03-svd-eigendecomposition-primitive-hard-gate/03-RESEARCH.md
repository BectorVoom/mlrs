# Phase 3: SVD / Eigendecomposition Primitive (Hard Gate) - Research

**Researched:** 2026-06-12
**Domain:** Iterative dense linear algebra (Jacobi SVD + symmetric eigendecomposition) in CubeCL 0.10 `#[cube]` kernels; first real ROCm/HIP runtime bring-up
**Confidence:** HIGH on ROCm bring-up + f64 finding (empirically built+ran on gfx1100), HIGH on existing-code integration, MEDIUM on Jacobi kernel mechanics (algorithm is textbook; CubeCL expression validated against in-repo patterns but not yet built), MEDIUM on convergence constants (must be tuned empirically against the D-08 sweep)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** TWO distinct routines — (1) general one-sided Jacobi **SVD** for arbitrary matrices; (2) a separate classic (two-sided) Jacobi **symmetric-eigendecomposition**. NOT one unified core. Two kernels, each on its own correctness surface.
- **D-02:** **Thin / economy extent.** SVD returns U (m×k), S (k), Vᵀ (k×n), `k = min(m,n)` — matching `numpy.linalg.svd(full_matrices=False)`. No full square U (m×m) / V (n×n).
- **D-03:** **Raw output; sign-align only at comparison.** Kernel returns whatever sign Jacobi produces. `mlrs-core/src/sign_flip.rs` (`align_rows`/`align_sign`) canonicalizes ONLY at oracle-comparison time. No device-side flip kernel.
- **D-04:** **Descending LAPACK/numpy ordering.** S[0] largest; eigenvalues descending. NB `np.linalg.eigh` returns ASCENDING — the eig primitive must sort descending on-device.
- **D-05:** **SVD handles tall AND wide** (`m ≥ n` and `m < n`). When `m < n`, run Jacobi on `Aᵀ` and swap U↔V. Shape passed explicitly as `(rows, cols)` per call; `DeviceArray` stays flat 1D.
- **D-06:** **Eig assumes square symmetric; trusts symmetry** — no `(A+Aᵀ)/2`. Only v1 feeder is the Phase-2 covariance primitive.
- **D-07 (CROSS-CUTTING):** GPU correctness gate moves wgpu → **cpu + rocm**, project-wide from Phase 3. ROCm 7.1.1, hipcc, gfx1100, `/dev/kfd` + `/dev/dri/renderD128` runnable. **ROCm/HIP bring-up is the FIRST task** — rocm has only ever compiled, never executed. *(See CRITICAL FINDING 2 below: the D-07 premise that "f64 runs natively on gfx1100" is empirically FALSE at the CubeCL layer — f64 is unsupported on the HIP backend in stock cubecl 0.10.)*
- **D-08:** Validation = random sweep (well-conditioned tall/wide/square, f32+f64, cpu+rocm) + degenerate cases (rank-deficient / repeated / zero singular values, near-identity, clustered-eigenvalue). Mostly small + one moderate ~256×64 case.
- **D-09:** Primary reference = committed numpy `.npz` fixtures (`np.linalg.svd(full_matrices=False)` / `np.linalg.eigh`) via Phase-1 `gen_oracle.py` + npz-loader, compared after `svd_flip`, PLUS reference-free algebraic invariants (reconstruction, orthonormality, eig residual). Host Jacobi REJECTED as primary.
- **D-10:** Hold global 1e-5 abs+rel; per-family looser bound ONLY if a real case can't hold it, and record which case forced it. Do not pre-loosen.
- **D-11:** Extend the Phase-2 build-failing PoolStats memory gate to SVD/eig with HARD assertions: (1) bounded Jacobi scratch (allocation count does NOT grow with sweep count; scratch from BufferPool, recycled); (2) eig reuses covariance/GEMM output buffer; (3) no host round-trip between sweeps (convergence loop device-resident, only final result read back).
- **D-12:** Convergence policy = fixed internal constants (off-diagonal-norm threshold + max-sweep cap), NOT public API. Researcher chooses constants that hold 1e-5.
- **D-13:** Feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>` in `mlrs-kernels`; launch wrappers + host orchestration in `mlrs-backend`. `assert_close` 1e-5; `thiserror` in libs / `anyhow` at boundaries; source/test separation per AGENTS.md (NO in-source `mod tests`).

### Claude's Discretion
- The entire **Jacobi rotation-kernel design** — one-sided vs two-sided mechanics, rotation-pair scheduling (round-robin / chess-tournament parallel ordering), plane/subgroup vs shared-memory expression. (This research makes those recommendations below.)
- Module/file layout within `mlrs-kernels` and `mlrs-backend` (`prims/svd.rs`, `prims/eig.rs` or similar) — honor source/test separation.
- Internal convergence constants (D-12), sweep ordering, block/tile sizes (subject to no-hardcoded-plane-width carried from Phase 2 D-03).
- Exact random shapes/seeds for the sweep, and which cases get committed `.npz` fixtures vs invariant-only checks (subject to D-08 + D-09 coverage).
- Naming of new primitive error variants (extend the `thiserror` enums).

### Deferred Ideas (OUT OF SCOPE)
- Per-estimator-family tolerance tables (activate only if a real SVD/eig case can't hold 1e-5, and only for that case).
- Unified single Jacobi core (eig derived from SVD) — rejected (D-01).
- Full U (m×m) / V (n×n) SVD — deferred (D-02 thin only).
- Device-side `svd_flip` kernel — deferred (D-03 aligns at comparison).
- Defensive eig symmetrization `(A+Aᵀ)/2` — deferred (D-06 trusts the covariance feeder).
- wgpu as a gate — dropped to opportunistic (D-07).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PRIM-05 | An SVD / eigendecomposition primitive (GPU Jacobi or equivalent) serves PCA, TruncatedSVD, and the OLS/Ridge SVD solver paths, validated against an oracle within tolerance | Two-routine Jacobi design (one-sided SVD + two-sided symmetric-eig) in §Architecture Patterns; thin-SVD extraction (§Pattern 3); wide-path Aᵀ-swap (§Pattern 4); D-08 sweep + D-09 fixtures/invariants in §Validation Architecture; convergence constants in §Common Pitfalls / §Open Questions; D-11 memory gate in §Validation Architecture; ROCm bring-up de-risked empirically in §Summary and §Environment Availability |
</phase_requirements>

## Summary

This phase has two independent risk surfaces: **(A) ROCm/HIP runtime bring-up** (the project's first real GPU execution) and **(B) hand-writing two iterative Jacobi kernels** (no pre-built CubeCL SVD primitive exists). Both were investigated; bring-up was de-risked empirically.

**ROCm bring-up was reproduced end-to-end during this research and it WORKS — but only after two concrete fixes, both discovered here.** A real `#[cube]` saxpy kernel was compiled to HIP and executed on gfx1100 with correct read-back (`test spike_saxpy_runs_on_active_backend ... ok`). The two fixes are: (1) `crates/mlrs-backend/src/runtime.rs:20` imports `cubecl::rocm::{RocmDevice, RocmRuntime}` which **does not exist** — the correct path is `cubecl::hip::{HipRuntime as ActiveRuntime, AmdDevice as ActiveDevice}`; (2) the `rocm` Cargo feature must additionally enable `cubecl`'s `std` (and `default`) feature, because `cubecl-hip` only compiles when the `multi_threading` cfg is active, and that cfg is `all(feature="std", not(wasm))` — with `default-features = false` on the workspace `cubecl`, `std` is never propagated to the HIP backend and `cubecl-hip` fails to compile (`unresolved import cubecl_runtime::stream::MultiStream`). Both fixes are small and verified.

**CRITICAL: the D-07 assumption that f64 runs natively on gfx1100 is empirically FALSE at the CubeCL layer.** `capability::supports_type(FloatKind::F64)` returns **false** on the HIP backend (measured: `capability backend=rocm f32_supported=true f64_supported=false`), and a real f64 GEMM oracle test **SKIPPED** on rocm via the existing `skip_f64_with_log` gate. Root cause: `cubecl-cpp 0.10.0` (the shared C++/HIP codegen used by the HIP backend) has `F64` **commented out** of `register_supported_types` (`src/shared/base.rs:2115`). This is a CubeCL library limitation, not a gfx1100 hardware limitation. **Consequence for the planner: f64 SVD/eig is validated on the `cpu` backend (which DOES run f64), and f64 on `rocm` continues to SKIP-with-log exactly as Phase 2.** This is strictly the safe behavior the existing capability gate already implements — but the phase's success criteria and ROADMAP wording must be reconciled: "f64 runs on rocm" is not achievable without patching CubeCL.

For the **Jacobi kernels**, the recommended design is: **one-sided Jacobi for SVD** and **classic two-sided cyclic Jacobi for symmetric-eig**, both as **single-cube, shared-memory kernels** with the matrix resident in `SharedMemory`, the convergence sweep loop entirely inside the kernel (satisfying D-11 gate 3 — no host round-trip between sweeps), and a **round-robin (chess-tournament) parallel rotation-pair schedule** so `floor(n/2)` column/index pairs rotate concurrently per step. This fits the v1 sizes (mostly small, one ~256×64) within a single cube's shared memory and thread budget, and reuses the exact `SharedMemory` + `sync_cube` + `PLANE_DIM`-aware idioms already proven in `mlrs-kernels/src/reduce.rs`.

**Primary recommendation:** Sequence the phase as **(1) ROCm bring-up + the two fixes above + reconcile the f64-on-rocm expectation → (2) one-sided Jacobi SVD kernel (tall path) → (3) thin-U extraction + descending sort + wide Aᵀ-swap path → (4) two-sided symmetric-eig kernel → (5) D-09 fixtures + invariants + D-11 memory gate.** Do task 1 before any SVD work; it is the single highest bring-up risk and it has known-required fixes.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Jacobi rotation sweep (off-diagonal annihilation) | `mlrs-kernels` `#[cube]` kernel | — | Device compute; feature-free generic-over-`F` (D-13) |
| Off-diagonal-norm convergence test | `mlrs-kernels` (in-kernel reduction) | — | Must stay device-resident between sweeps (D-11 gate 3) |
| Sweep-loop orchestration / launch config | `mlrs-backend` `prims/svd.rs` `prims/eig.rs` | — | Owns `ActiveRuntime`, launch wrappers, shape validation (D-13) |
| Thin-U extraction (column-normalize A·V) | `mlrs-kernels` + `mlrs-backend` | reuses Phase-2 GEMM | Algebraic post-step; GEMM already validated |
| Descending sort of S / eigenpairs | `mlrs-kernels` (small-n on-device) or `mlrs-backend` host permute | — | k=min(m,n) is small in v1; either is acceptable (see Pattern 5) |
| Wide-matrix Aᵀ-and-swap | `mlrs-backend` host orchestration | reuses GEMM transpose flags (D-06) | Pure dispatch + label swap; no new kernel |
| Sign alignment (svd_flip) | `mlrs-core/src/sign_flip.rs` (test-time only) | — | D-03: applied at oracle comparison, NOT in kernel |
| Memory gate (PoolStats assertions) | `mlrs-backend/tests/memory_gate_test.rs` (extend) | `pool.rs` counters | D-11 extends the Phase-2 build-failing gate |
| numpy reference fixtures | `scripts/gen_oracle.py` (build-time) + `mlrs-core` npz loader | — | D-09; hermetic committed blobs |

## Standard Stack

### Core

No new external dependencies are introduced by this phase. The Jacobi kernels are hand-written `#[cube]` code; all supporting machinery already exists in the workspace.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 | `#[cube]` kernels, `SharedMemory`, `sync_cube`, plane ops, launch | Project-mandated device-kernel layer (workspace pin) [VERIFIED: workspace Cargo.toml] |
| `cubecl-hip` | 0.10.0 (transitively, via `cubecl/rocm` → `hip`) | HIP/ROCm runtime backend | The only ROCm path in cubecl 0.10; `rocm = ["hip"]` [VERIFIED: cubecl 0.10.0 Cargo.toml + built locally] |
| `cubecl-hip-sys` | 7.1.5280200 (matches ROCm 7.1.1) | HIP FFI bindings | Auto-selected by `hipconfig`-reported patch `7.1.52802` [VERIFIED: hipconfig + registry] |
| `cubek-matmul` / `cubek-std` | 0.2.0 | GEMM substrate (reused for A·V, reconstruction, residuals) | Already wired in Phase 2 (PRIM-01) [VERIFIED: backend Cargo.toml] |
| `bytemuck` / `thiserror` | 1 / 2 | Pod cast for read-back; typed error enums | Carried from Phases 1–2 [VERIFIED: workspace Cargo.toml] |
| `npyz` | 0.9 (npz) | `.npz` fixture loader for D-09 | Carried from Phase 1 oracle infra [VERIFIED: workspace Cargo.toml] |

### Supporting

| Component | Location | Purpose | When to Use |
|-----------|----------|---------|-------------|
| `reduce.rs` plane/shared dual-path | `mlrs-kernels/src/reduce.rs` | Pattern template for the in-kernel off-diagonal-norm reduction | Convergence test inside the sweep |
| `gemm.rs` (transpose flags D-06) | `mlrs-backend/src/prims/gemm.rs` | A·V (thin-U), Aᵀ-swap, reconstruction & residual invariants | Post-sweep extraction + test invariants |
| `covariance.rs` | `mlrs-backend/src/prims/covariance.rs` | The eig path's only feeder (square symmetric Gram, D-06) | Eig test inputs + buffer-reuse target (D-11 gate 2) |
| `sign_flip.rs` | `mlrs-core/src/sign_flip.rs` | `align_rows`/`align_sign` at comparison time (D-03) | Oracle compare, NOT in kernel |
| `pool.rs` / `device_array.rs` | `mlrs-backend/src` | `BufferPool` + `PoolStats` + `release_into` + `to_host_metered` | D-11 bounded-scratch gate |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| One-sided Jacobi SVD | Two-sided Jacobi on AᵀA, then back-substitute | AᵀA squares the condition number → worse f32 accuracy on the clustered/rank-deficient D-08 cases; one-sided operates on A directly and is the standard accurate choice. Use one-sided. |
| Single-cube shared-memory sweep | Multi-cube global-memory sweep with host-driven sweep loop | Multi-cube needs a host round-trip per sweep (violates D-11 gate 3) OR device-wide barriers cubecl 0.10 does not expose. Single-cube keeps the loop in-kernel. v1 sizes (≤256×64) fit one cube. Use single-cube; flag larger sizes as a future block-Jacobi item. |
| Round-robin parallel pair schedule | Sequential cyclic-by-row Jacobi | Sequential is simpler but serializes rotations (one pair per step); round-robin rotates `floor(n/2)` disjoint pairs per step, matching the GPU's parallel units. Use round-robin; keep a sequential reference in the host-side test oracle is NOT needed (D-09 uses numpy + invariants). |
| f64 validated on rocm | f64 validated on cpu only (rocm skips f64) | Forced by CubeCL's HIP backend not registering F64 (see CRITICAL FINDING 2). cpu runs f64; rocm runs f32. This is the only viable split without patching cubecl. |

**Installation:** No new packages. The only build-config change is the `rocm` feature wiring (see Pattern 1).

**Version verification:** Performed during research:
- `cubecl-hip 0.10.0`, `cubecl-hip-sys 7.1.5280200` and `7.2.5321100` both present in the local registry [VERIFIED: `~/.cargo/registry`]. `hipconfig` reports patch `7.1.52802` → maps to `7.1.5280200`. Note Cargo's resolver initially pulled `7.2.5321100`; the build still succeeded after the std-feature fix, so the sys-crate minor skew was NOT the blocker.
- ROCm 7.1.1, `hipcc`/`hipconfig` on PATH at `/opt/rocm/bin`, `libamdhip64.so.7.1.70101` + `libhiprtc.so.7.1.70101` present [VERIFIED: filesystem].

## Package Legitimacy Audit

> No external packages are installed by this phase. All crates are already pinned and built in Phases 1–2. slopcheck not required (no new registry installs). The transitive `cubecl-hip` / `cubecl-hip-sys` crates are pulled by the existing `cubecl` pin and were compiled successfully locally during this research.

| Package | Registry | Age | Source Repo | Disposition |
|---------|----------|-----|-------------|-------------|
| cubecl-hip | crates.io | published (0.10.0 line) | github.com/tracel-ai/cubecl | Approved — transitive via existing `cubecl` pin; built locally |
| cubecl-hip-sys | crates.io | published | github.com/tracel-ai/cubecl | Approved — transitive; matches ROCm 7.1.1 |

**Packages removed due to slopcheck [SLOP] verdict:** none (no new installs)
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────────┐
   host (mlrs-backend)   │  prims/svd.rs  /  prims/eig.rs               │
                         │                                             │
  DeviceArray A ───────► │  validate (rows,cols) ──► [m<n? transpose    │
  (rows,cols)            │                            label-swap, D-05] │
                         │         │                                   │
                         │         ▼  acquire scratch from BufferPool   │
                         │   ┌──────────────────────────────────────┐  │
                         │   │  LAUNCH jacobi_*_kernel (1 cube)      │  │
                         └───┼──────────────────────────────────────┼──┘
                             │  SharedMemory<F> A_tile (resident)    │
                             │   ┌────────── sweep loop ──────────┐  │   device-resident
                             │   │ round-robin pair schedule      │  │   (NO host round-trip
                             │   │  ▶ compute rotation (c,s)      │  │    between sweeps —
                             │   │  ▶ apply to col pair (A·V acc) │  │    D-11 gate 3)
                             │   │  ▶ sync_cube                   │  │
                             │   │  ▶ off-diag-norm reduction     │  │
                             │   │  ▶ if norm<thresh OR sweep>cap │  │
                             │   │       break  (D-12 constants)  │  │
                             │   └────────────────────────────────┘  │
                             │   write columns → global out handles  │
                             └───────────────┬───────────────────────┘
                                             │
   host post-step                            ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ thin-U: S = colnorm(A·V); U = (A·V)/S   (reuse Phase-2 GEMM, D-02) │
   │ descending sort of S (+ permute U cols, Vᵀ rows)  (D-04)           │
   │ [m<n: swap U↔V back]                                               │
   └──────────────────────────────────────────────────────────────────┘
                                             │
   read-back (terminal, metered)             ▼
   U (m×k), S (k), Vᵀ (k×n)  ──► test: svd_flip align ──► numpy fixture compare (D-09)
                              └─► invariants: ‖UΣVᵀ−A‖, ‖UᵀU−I‖, ‖VᵀV−I‖  (D-09)
```

### Recommended Project Structure

```
crates/mlrs-kernels/src/
├── jacobi_svd.rs        # one-sided Jacobi SVD #[cube(launch)] kernel(s)
├── jacobi_eig.rs        # two-sided symmetric-eig #[cube(launch)] kernel(s)
└── lib.rs               # pub mod jacobi_svd; pub mod jacobi_eig;

crates/mlrs-backend/src/prims/
├── svd.rs               # host: validate, transpose-swap (D-05), launch, thin-U, sort
├── eig.rs               # host: validate-square, launch, descending sort (D-04)
└── mod.rs               # pub mod svd; pub mod eig;

crates/mlrs-backend/tests/
├── svd_test.rs          # D-09 fixtures + invariants, f32 (rocm+cpu) / f64 (cpu)
├── eig_test.rs          # D-09 eigh fixtures + eig-residual invariant
└── memory_gate_test.rs  # EXTEND with the 3 D-11 SVD/eig assertions

scripts/gen_oracle.py    # EXTEND: np.linalg.svd(full_matrices=False), np.linalg.eigh
tests/fixtures/          # new committed: svd_*_seedNN.npz, eigh_*_seedNN.npz
```

### Pattern 1: ROCm/HIP runtime bring-up (D-07, FIRST TASK — known fixes)

**What:** Make `--features rocm` compile AND run a trivial kernel on gfx1100.
**When to use:** Task 1, before any SVD work.

Two required edits (both discovered + verified during this research):

```rust
// crates/mlrs-backend/src/runtime.rs  — FIX the non-existent module path.
// WRONG (current line 20): cubecl::rocm::{RocmDevice, RocmRuntime}  ← no such module
// cubecl 0.10 re-exports `cubecl_hip as hip` under the `hip` feature (rocm = ["hip"]);
// the device struct is `AmdDevice` (no `HipDevice` alias exists).
#[cfg(feature = "rocm")]
pub use cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime};
```

```toml
# crates/mlrs-backend/Cargo.toml  — rocm must enable cubecl's std/default features.
# Root cause: cubecl-hip only compiles when the `multi_threading` cfg is set, and that
# cfg = all(feature="std", not(wasm)). With workspace `cubecl { default-features=false }`,
# `cubecl/std` is NOT propagated to cubecl-hip, so its `use cubecl_runtime::stream::
# MultiStream` import is unresolved (the `event` module is `#[cfg(multi_threading)]`).
rocm = ["cubecl/rocm", "cubecl/std", "cubecl/default"]
```

*(`cubecl/default` pulls `cubecl-hip?/default` → `cubecl-runtime/default` → `std`. Minimally `cubecl/std` plus `cubecl-hip/default` would suffice; `cubecl/default` is the simplest correct switch and was the one verified.)*

Acceptance for task 1 (all verified reproducible during research):
- `cargo build -p mlrs-backend --no-default-features --features rocm` succeeds.
- `cargo test -p mlrs-backend --no-default-features --features rocm --test spike_test spike_saxpy_runs_on_active_backend` passes (real HIP kernel on gfx1100).
- `capability::active_backend_name() == "rocm"`, `supports_type(F32) == true`.
- **Expected and correct:** `supports_type(F64) == false` on rocm → f64 oracle tests SKIP-with-log (see Pitfall 1). Do NOT treat the f64 skip on rocm as a bug.

### Pattern 2: One-sided Jacobi SVD, single-cube shared-memory kernel (PRIM-05 core)

**What:** Orthogonalize the columns of A by a sequence of plane rotations applied on the right (`A ← A·J`), accumulating the rotations into V. On convergence, the column norms of the rotated A are the singular values and the normalized columns are U.

**When to use:** SVD of `m × n` with `m ≥ n` (the wide path routes here via Aᵀ, Pattern 4).

Algorithm (textbook one-sided / Hestenes Jacobi):
1. Load A (m×n) into `SharedMemory`; initialize V = I in shared (n×n), or accumulate V implicitly.
2. **Sweep:** for each disjoint column pair (i,j) in the round-robin schedule (Pattern 6):
   - compute `α = Σ a_ki²`, `β = Σ a_kj²`, `γ = Σ a_ki·a_kj` (dot products over the m rows);
   - if `|γ|` is below the off-diagonal threshold, skip this pair (no rotation);
   - else compute the Jacobi rotation `(c, s)` that zeroes `γ`:
     `ζ = (β − α)/(2γ); t = sign(ζ)/(|ζ| + sqrt(1+ζ²)); c = 1/sqrt(1+t²); s = c·t;`
   - apply to columns i and j of A (and of V): `a_ki' = c·a_ki − s·a_kj`, `a_kj' = s·a_ki + c·a_kj`.
   - `sync_cube()` after each parallel rotation step.
3. **Convergence test (in-kernel):** after each full sweep, reduce the off-diagonal Frobenius norm `sqrt(Σ_{i≠j} γ_ij²)` (or max `|γ_ij|`); break when below threshold or sweep count exceeds the cap (D-12).
4. Write rotated A columns (= U·diag(S) unnormalized) and accumulated V to global out handles. Host extracts S + U (Pattern 3).

Why one-sided (not two-sided-on-AᵀA): one-sided works on A directly, so it does **not** square the condition number — essential for the f32 clustered-eigenvalue / rank-deficient D-08 cases to hold 1e-5.

Why single-cube shared-memory: the whole convergence loop must run **inside one kernel launch** with no host round-trip between sweeps (D-11 gate 3). cubecl 0.10 has cube-scoped `sync_cube()` but no portable device-wide barrier, so a multi-cube sweep would force a host-driven loop (a read-back per sweep). A single cube holding the matrix in `SharedMemory` keeps the loop device-resident. v1 sizes (mostly small, one ~256×64) fit: 256×64 f32 ≈ 64 KiB, within gfx1100's 64 KiB LDS per workgroup — **verify LDS budget during task 2** and document.

### Pattern 3: Thin-U + S extraction without forming square U (D-02)

**What:** Recover U (m×k), S (k) from the converged one-sided result.

After convergence, the columns of the rotated A are mutually orthogonal: column j has 2-norm `σ_j` and direction `u_j`. So:
- `S[j] = ‖(A·V)[:,j]‖₂` (column L2 norm — reuse the Phase-2 row/col L2-norm reduction);
- `U[:,j] = (A·V)[:,j] / S[j]` (column-normalize; guard `S[j] ≈ 0` for zero/rank-deficient singular values — set U column to 0 or leave the unnormalized direction, matching numpy's behavior for exact-zero singular values; numpy returns an orthonormal completion, so for the **rank-deficient D-08 case** prefer the **invariant checks** over exact fixture match on the null-space columns — see Pitfall 4).

This yields the thin factors directly; the full square U (m×m) is never formed (D-02). Vᵀ is the transpose of the accumulated V (k×n after thinning to the first k columns).

### Pattern 4: Wide-matrix path via Aᵀ-and-swap (D-05)

**What:** Handle `m < n` by running the tall kernel on Aᵀ.

`A = U Σ Vᵀ  ⇒  Aᵀ = V Σ Uᵀ`. So compute the SVD of `Aᵀ` (which is `n × m`, tall since `n > m`) with the Pattern-2 kernel, obtaining `(U', S', V'ᵀ)` of Aᵀ, then **swap**: `U = V'`, `S = S'`, `Vᵀ = U'ᵀ`. Pure host-side label swap + a transpose of the input (reuse the Phase-2 GEMM transpose flag D-06, or upload Aᵀ via shape swap — no materialized transpose buffer needed). This is dispatch logic in `prims/svd.rs`, no new kernel.

**Tolerance confidence:** the Aᵀ-swap is exact algebra (no extra arithmetic beyond one transpose), so it holds the same 1e-5 the tall path holds — **MEDIUM confidence pending the task-3 wide-case oracle run**; if it fails, the fallback is a dedicated wide kernel, but this is not expected (Open Question 4).

### Pattern 5: Two-sided classic Jacobi symmetric-eig + descending sort (D-01, D-04, D-06)

**What:** For a square symmetric A (the covariance Gram), apply two-sided rotations `A ← Jᵀ A J` to drive off-diagonal entries to zero; the diagonal converges to eigenvalues, the accumulated J columns are eigenvectors.

Differences from the SVD kernel:
- rotation is applied on **both** sides (rows AND columns) of A;
- the rotation angle zeroes the symmetric off-diagonal `a_ij`: `θ = (a_jj − a_ii)/(2 a_ij)` then the same `t, c, s` formula;
- trusts symmetry (D-06) — no `(A+Aᵀ)/2`;
- eigenvalues are the converged diagonal; **sort descending** and permute eigenvector columns accordingly (D-04). NB `np.linalg.eigh` returns **ascending**, so the fixture must be reversed OR the on-device sort produces descending and the test reverses the numpy reference — pick one and document. Recommendation: sort descending on-device (estimators inherit the right order per D-04) and reverse the numpy `eigh` arrays in the test comparison.

Descending sort: for v1 `k = n ≤ ~64`, a simple selection/insertion sort is fine — do it **on the host after read-back is acceptable for the eig diagonal**, OR in-kernel with a single thread; both hold 1e-5 (sorting is exact). Host-side sort is simpler and keeps the kernel focused; the D-11 "device-resident" gate concerns the *convergence loop*, not the final O(n) sort of an already-converged result (only the final result is read back — gate 3 still holds).

### Pattern 6: Round-robin (chess-tournament) parallel rotation-pair schedule

**What:** Schedule `floor(n/2)` disjoint column/index pairs to rotate concurrently per step, cycling through all `n(n-1)/2` pairs over `n−1` steps (one full sweep).

Standard "round-robin tournament" pairing: fix index 0, rotate the others. For n even, n−1 rounds cover all pairs; for n odd, add a bye. Express in `#[cube]` with a comptime-unrolled or `while`-driven schedule over the round index, each unit/plane handling one pair's dot products + rotation. This is the parallel-friendly ordering (disjoint pairs touch disjoint columns, so rotations within a step are independent — no write conflict). Use `sync_cube()` between steps.

CubeCL expression notes (verified against repo patterns + manuals):
- `continue` is **NOT supported** in `#[cube]` (loop_control manual) — use `if`-wrapping to skip below-threshold pairs.
- `SharedMemory::<F>::new(SIZE)` requires a **compile-time** `usize` size — size the tile to the **max** supported n (e.g. a comptime cap like 256) and bound the active region by a runtime `n`, exactly as `reduce.rs` sizes `SharedMemory::<F>::new(256usize)` and guards with `input.len()`.
- Use `F::from_int(i)` / `F::cast_from(lit)` for generic constants (algebra manual); `.sqrt()`, `.abs()` require the `Float` bound (already in `<F: Float + CubeElement>`).
- Do **not** hardcode the plane width — use `PLANE_DIM` if a plane-path dot product is used (carried Phase-2 D-03). For the single-cube design, a `SharedMemory` tree reduction (as in `reduce_sumsq_shared`) is the simplest correct off-diagonal-norm path and avoids plane-width portability concerns.

### Anti-Patterns to Avoid

- **Two-sided Jacobi on AᵀA for the SVD path:** squares the condition number; fails f32 clustered/rank-deficient cases. Use one-sided on A.
- **Host-driven sweep loop (read-back per sweep):** violates D-11 gate 3. Keep the loop in-kernel (single cube).
- **Hardcoding plane width / 32:** breaks portability (Phase-2 D-03). Use `PLANE_DIM` or a shared-memory tree.
- **Materializing a transpose buffer for the wide path:** reuse GEMM transpose flags / shape-swap (D-06).
- **Device-side svd_flip:** D-03 forbids it — align only at comparison.
- **Assuming f64 runs on rocm:** it does not (CRITICAL FINDING 2). Gate f64 to cpu.
- **`mod tests` in source files:** AGENTS.md §2 — tests live in `tests/`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense matmul for A·V, reconstruction, residuals | A bespoke matmul kernel | Phase-2 `gemm()` (cubek-matmul) | Already validated 1e-5 on cpu+rocm; transpose flags D-06 |
| Column / row L2 norms (S extraction, orthonormality) | A new norm kernel | Phase-2 `reduce` sumsq/L2 dual-path | Validated, plane+shared paths exist |
| Sign canonicalization | A flip kernel | `sign_flip::align_rows` at comparison (D-03) | Already implemented, FOUND-08 |
| numpy reference generation | Live sklearn in tests | `gen_oracle.py` committed `.npz` (D-09) | Hermetic, no Python at test time (D-03) |
| f64 capability decision on rocm | New gating logic | `skip_f64_with_log()` | Already returns the correct skip on rocm |
| Buffer reuse / scratch metering | A new pool | `BufferPool` + `release_into` + `PoolStats` | D-11 gate asserts on these counters |

**Key insight:** This phase is almost entirely *new kernel logic on top of already-validated primitives*. The only genuinely novel device code is the two Jacobi sweep kernels; everything around them (matmul, norms, sign, fixtures, pool) is reused. Resist re-implementing any of the reused pieces inside the Jacobi kernels.

## Runtime State Inventory

> Not a rename/refactor/migration phase. The only "runtime state" concern is the build-config / runtime-resolution change for ROCm, captured below for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys involved. | None |
| Live service config | None. | None |
| OS-registered state | None. | None |
| Secrets/env vars | `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set in the environment (observed via `hipconfig`); gfx1100 is native RDNA3 so the override is benign but present. `ROCM_PATH`/`HIP_PATH` are **unset** — `cubecl-hip-sys` build.rs falls back to `hipconfig` on PATH, which works. | Document: if a clean shell lacks `/opt/rocm/bin` on PATH, the HIP build fails to find `hipconfig`; the build task should ensure `/opt/rocm/bin` is on PATH (it currently is). |
| Build artifacts | The `runtime.rs` `cubecl::rocm` path + the `rocm` feature wiring are **stale/incorrect** (never executed before this phase). | Apply the two Pattern-1 fixes (verified). |

## Common Pitfalls

### Pitfall 1: Expecting f64 to run on rocm (D-07 premise is false at the CubeCL layer)
**What goes wrong:** Tests assert f64 SVD results on rocm; they SKIP (via `skip_f64_with_log`) and the gate appears to "pass" without exercising f64, OR a planner writes a task expecting f64 numbers from gfx1100.
**Why it happens:** `cubecl-cpp 0.10.0` comments out `F64` in `register_supported_types` (`src/shared/base.rs:2115`), so the HIP backend reports `supports_type(F64) == false` regardless of gfx1100 hardware.
**How to avoid:** Validate f64 SVD/eig on the **cpu** backend (which runs f64 — verified: `gemm_f64_matches_host_ref ... ok` on cpu). On rocm, validate **f32** and let f64 skip-with-log. Make the f64 split explicit in every SVD/eig oracle test (mirror the existing `gemm_test.rs` skip pattern). Reconcile ROADMAP/PROJECT/CONTEXT wording: "f64 runs on rocm" → "f64 runs on cpu; rocm runs f32; f64-on-rocm blocked by cubecl-cpp F64 registration."
**Warning signs:** CI log line `gemm f64 backend=rocm: SKIPPED`. That is correct, not a failure — but it means f64 coverage comes from cpu only.

### Pitfall 2: ROCm build fails with `unresolved import cubecl_runtime::stream::MultiStream`
**What goes wrong:** `cargo build --features rocm` fails compiling `cubecl-hip` with 3 unresolved imports (`MultiStream`, `ResolvedStreams`, `EventStreamBackend`).
**Why it happens:** Those symbols live in `cubecl-runtime`'s `stream::event` module, gated `#[cfg(multi_threading)]` = `all(feature="std", not(wasm))`. With workspace `cubecl { default-features=false }` and `rocm = ["cubecl/rocm"]`, the `std` feature is never enabled on `cubecl-runtime`, so the module is absent but `cubecl-hip` imports it unconditionally.
**How to avoid:** `rocm = ["cubecl/rocm", "cubecl/std", "cubecl/default"]` (Pattern 1). Verified to fix the compile.
**Warning signs:** Three `E0432` errors all pointing into `cubecl-hip-0.10.0/src/compute/`.

### Pitfall 3: f32 clustered / repeated-eigenvalue accuracy (the likely 1e-5 stressor)
**What goes wrong:** Singular vectors / eigenvectors for clustered or repeated eigenvalues are ill-conditioned — the *subspace* is well-defined but individual vectors rotate freely, so a fixture comparison (even after svd_flip) can exceed 1e-5 in f32.
**Why it happens:** When σ_i ≈ σ_j, the rotation that separates them is governed by a near-zero denominator; f32 cancellation makes the chosen basis within the degenerate subspace effectively arbitrary.
**How to avoid:** For the clustered/repeated/rank-deficient D-08 cases, rely on the **reference-free invariants** (reconstruction ‖UΣVᵀ−A‖, orthonormality ‖UᵀU−I‖) which are basis-invariant, rather than exact eigenvector fixture match. Reserve exact-fixture comparison for well-conditioned cases. If a *well-conditioned* case still can't hold 1e-5 in f32, that — and only that — is the documented trigger to introduce a per-family looser bound (D-10), recording which case forced it.
**Warning signs:** Reconstruction invariant passes but per-vector fixture compare fails on the clustered case → it's a basis-degeneracy artifact, not an arithmetic bug.

### Pitfall 4: Zero / rank-deficient singular values and U column-normalization
**What goes wrong:** Pattern-3 thin-U divides by `S[j]`; for an exact-zero singular value this is a divide-by-zero, and numpy fills those U columns with an orthonormal completion that your kernel won't reproduce.
**Why it happens:** A rank-deficient matrix has σ_j = 0 for j ≥ rank; the rotated column is the zero vector with no defined direction.
**How to avoid:** Guard `S[j]` against a near-zero floor (reuse the `NEAR_ZERO_FLOOR` concept from Phase-1 `assert_close`); set the U column to 0 (or any unit vector) for σ_j ≈ 0. Validate the rank-deficient case with the **reconstruction invariant** (the zero columns contribute nothing to UΣVᵀ) and the **non-zero** singular values against the fixture, NOT the null-space U columns.
**Warning signs:** NaN/Inf in U columns; reconstruction passes but U orthonormality fails only on trailing columns.

### Pitfall 5: Convergence cap too low (silent non-convergence) or threshold too tight (never breaks)
**What goes wrong:** A max-sweep cap that's too small returns a not-fully-rotated result (off-diagonals not annihilated → reconstruction fails); a threshold tighter than f32 can reach loops to the cap every time (slow, and f32 can't reach an f64-tight threshold).
**How to avoid:** Threshold should be relative to the matrix scale and dtype epsilon (see Open Question 3 / convergence constants). cyclic Jacobi converges quadratically; ~10–15 sweeps suffice for n ≤ 256 (cuML's cuSolver Jacobi default is `n_iterations = 15`, `tol = 0` ≈ machine eps — a good reference point). Recommendation: threshold = `c · ε_F · ‖A‖_F` with a per-dtype `ε_F` (f32 ≈ 1.2e-7, f64 ≈ 2.2e-16), cap = 30 sweeps (generous; quadratic convergence means it rarely runs past ~12). Tune empirically in task 2/5 against the D-08 sweep and **record the final constants** in `jacobi_svd.rs` with a comment citing the case that fixed them.
**Warning signs:** Reconstruction invariant fails by a margin that shrinks if you raise the cap → cap too low. Tests time out / always hit cap → threshold unreachable for the dtype.

## Code Examples

Verified patterns from in-repo sources and the CubeCL manuals.

### In-kernel shared-memory reduction (template for the off-diagonal-norm convergence test)
```rust
// Source: mlrs-kernels/src/reduce.rs (reduce_sumsq_shared) — proven on cpu+wgpu, runs on rocm.
// The off-diagonal Frobenius-norm test is a sum-of-squares of the off-diagonal entries,
// reduced inside the cube with a log2 tree + sync_cube — no host round-trip (D-11 gate 3).
#[cube(launch)]
pub fn reduce_sumsq_shared<F: Float + CubeElement>(input: &Array<F>, output: &mut Array<F>) {
    let mut shared = SharedMemory::<F>::new(256usize);   // compile-time size; runtime-bounded
    let tid = UNIT_POS_X;
    let v = if (ABSOLUTE_POS_X as usize) < input.len() { input[ABSOLUTE_POS_X as usize] }
            else { F::from_int(0i64) };
    shared[tid as usize] = v * v;
    sync_cube();
    let mut s = CUBE_DIM_X / 2u32;
    while s > 0u32 { if tid < s { let val = shared[(tid + s) as usize]; shared[tid as usize] += val; }
                    sync_cube(); s /= 2u32; }
    if tid == 0u32 { output[CUBE_POS_X as usize] = shared[0usize]; }
}
```

### Generic-constant + Float-method usage inside a `#[cube]` rotation
```rust
// Source: Cubecl_algebra.md — generic constants and Float methods.
// Jacobi rotation (c,s) computation, generic over F:
let two   = F::from_int(2);
let one   = F::from_int(1);
let zeta  = (beta - alpha) / (two * gamma);
let t     = zeta.abs().sqrt(); // illustrative; real: sign(zeta)/(|zeta| + sqrt(1+zeta^2))
let c     = one / (one + t * t).sqrt();
let s     = c * t;
// continue is NOT supported (Cubecl_loop_control.md) — wrap skip in `if |gamma| > thresh { ... }`.
```

### ROCm device/runtime resolution (the verified bring-up path)
```rust
// Source: cubecl-hip-0.10.0/src/lib.rs (pub use runtime::HipRuntime; pub use device::*;)
//         + cubecl-0.10.0/src/lib.rs (#[cfg(feature="hip")] pub use cubecl_hip as hip;)
#[cfg(feature = "rocm")]
pub use cubecl::hip::{AmdDevice as ActiveDevice, HipRuntime as ActiveRuntime};
// AmdDevice derives Default (device.rs) → AmdDevice::default() works in active_client().
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| cpu + wgpu correctness gate | cpu + rocm gate (wgpu opportunistic) | This phase (D-07) | rocm runs f32 real GPU; f64 stays on cpu (CubeCL F64 gap) |
| `cubecl::rocm::*` (assumed in runtime.rs) | `cubecl::hip::*` (`HipRuntime`, `AmdDevice`) | cubecl 0.10 (`rocm = ["hip"]`) | runtime.rs must be fixed before rocm compiles |
| LAPACK divide-and-conquer SVD (cuML cuSolver path) | hand-written one-sided Jacobi `#[cube]` | This phase (no cubecl SVD primitive) | numerical contract is numpy/LAPACK agreement, not cuML kernel parity |

**Deprecated/outdated:**
- `cubecl::rocm` module: does not exist in 0.10 — use `cubecl::hip`.
- The premise "gfx1100 f64 native → f64 runs on rocm": true for hardware, false for cubecl 0.10's HIP backend (F64 unregistered in `cubecl-cpp`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A single cube holding a 256×64 f32 tile (≈64 KiB) fits gfx1100's LDS budget for the one-sided sweep | Pattern 2 | If LDS overflows, the moderate D-08 case needs a block-Jacobi or a smaller tile; tall-skinny still fits. **Verify LDS in task 2** (`rocminfo` reports LDS size; measure actual occupancy). |
| A2 | Round-robin parallel rotations within a step are write-conflict-free (disjoint columns) and converge in ~10–15 sweeps for n ≤ 256 | Pattern 6, Pitfall 5 | If convergence is slower for the clustered case, raise the cap; quadratic convergence makes >30 unlikely. Tune empirically. |
| A3 | The Aᵀ-and-swap wide path holds 1e-5 (exact algebra, one transpose) | Pattern 4 | If a wide case fails, a dedicated wide kernel is the fallback (Open Question 4); not expected. |
| A4 | Host-side descending sort of the converged eig diagonal does not violate D-11 gate 3 (only the final result is read back; the sort is post-convergence) | Pattern 5 | If a reviewer reads gate 3 as "no host involvement at all," do the sort in-kernel (single thread). Low risk. |
| A5 | cuML's `n_iterations=15`, `tol=0` (machine-eps) Jacobi defaults are a reasonable starting point for the D-12 constants | Pitfall 5, Open Q3 | These are cuSolver defaults for a different algorithm variant; treat as a starting estimate, tune against D-08. |
| A6 | `cubecl/default` on the rocm feature does not pull a *second* conflicting backend runtime | Pattern 1 | `default` includes `stdlib` + optional-default of each backend, but optional deps stay off unless their feature is on; verified the build links only the HIP runtime. Low risk. |

## Open Questions

1. **Final D-12 convergence constants (off-diagonal threshold + max-sweep cap) for f32 and f64.**
   - What we know: cyclic Jacobi converges quadratically; cuML cuSolver uses `n_iterations=15`, `tol≈eps`. A scale-relative threshold `c·ε_F·‖A‖_F` is the right form.
   - What's unclear: the exact `c` and cap that hold 1e-5 across *all* D-08 cases (especially clustered f32).
   - Recommendation: implement with threshold `= 8·ε_F·‖A‖_F` (ε_f32≈1.2e-7, ε_f64≈2.2e-16) and cap = 30; run the D-08 sweep in task 2/5; tighten/loosen and **record the forcing case** in a kernel comment. This is a tuning task, not a research blocker.

2. **gfx1100 LDS budget for the moderate 256×64 case (A1).**
   - What we know: gfx1100 (RDNA3) has 64 KiB LDS per workgroup; 256×64 f32 ≈ 64 KiB for A alone, plus V (n×n=64×64=16 KiB) + scratch.
   - What's unclear: whether A + V + reduction scratch fit one cube, or whether the moderate case needs the matrix in global memory with shared only for the active column pair tiles.
   - Recommendation: in task 2, measure with `rocminfo`/occupancy; if it overflows, keep A in **global** memory and stage only the active column pair into shared (the round-robin schedule touches `floor(n/2)` pairs — stage per-pair). Small cases stay fully shared. Document the chosen layout.

3. **f64-on-rocm reconciliation (process, not technical).**
   - What we know: cubecl-cpp 0.10 does not register F64 for the HIP/CUDA C++ backend; f64 skips on rocm.
   - What's unclear: whether the project accepts "f64 on cpu only" for v1, or wants to carry a cubecl patch (out of scope, high cost).
   - Recommendation: accept f64-on-cpu / f32-on-rocm for v1; update ROADMAP/PROJECT/CONTEXT D-07 wording. The orchestrator/planner owns the doc reconciliation (already flagged in CONTEXT as a reconciliation action).

4. **Wide-path correctness (A3).**
   - What we know: Aᵀ-swap is exact algebra.
   - What's unclear: empirical 1e-5 confirmation on a real wide case.
   - Recommendation: include a wide case in the task-3 oracle run; if it passes (expected), close this; if not, scope a dedicated wide kernel.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| ROCm toolkit | rocm gate (D-07) | ✓ | 7.1.1 | — |
| `hipcc` / `hipconfig` | cubecl-hip-sys build.rs | ✓ | HIP 7.1.52802 (`/opt/rocm/bin`) | — |
| `/dev/kfd` | HIP device access | ✓ | char dev, group `render` | — |
| `/dev/dri/renderD128` | HIP device access | ✓ | char dev, group `render` | — |
| `render` group membership | device access | ✓ | user is in `render` (gid 992) | — |
| `libamdhip64` / `libhiprtc` | HIP runtime link | ✓ | 7.1.70101 (`/opt/rocm/lib`) | — |
| AMD gfx1100 (RDNA3) | rocm kernel execution | ✓ | gfx1100 (verified via rocminfo + saxpy run) | — |
| f64 on rocm | f64 SVD/eig on gfx1100 | ✗ | unsupported (cubecl-cpp F64 unregistered) | **cpu backend runs f64** |
| cpu backend (CubeCL CPU runtime) | f64 validation + portable gate | ✓ | cubecl 0.10 cpu | — |
| Python + numpy (`/tmp` venv) | regenerate D-09 fixtures (build-time only) | per Phase-1/2 (PEP 668 venv) | — | fixtures are committed blobs; venv only for regen |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** f64 on rocm → use the **cpu** backend for f64 SVD/eig validation (rocm validates f32). This is already the behavior the `skip_f64_with_log` gate produces.

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`). This section drives VALIDATION.md.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (integration tests in `tests/`, AGENTS.md §2 — no in-source `mod tests`) |
| Config file | none — `cargo test` with per-feature backend selection |
| Quick run command | `cargo test -p mlrs-backend --no-default-features --features cpu --test svd_test` |
| Full suite (cpu) | `cargo test -p mlrs-backend --no-default-features --features cpu` |
| Full suite (rocm) | `cargo test -p mlrs-backend --no-default-features --features rocm --features cubecl/std --features cubecl/default` |

*(Note the rocm invocation requires the Pattern-1 feature additions; once the `rocm` feature in Cargo.toml is fixed to include `cubecl/std`+`cubecl/default`, the plain `--features rocm` form works.)*

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRIM-05 | ROCm bring-up: real kernel runs on gfx1100 | smoke | `cargo test -p mlrs-backend --no-default-features --features rocm --features cubecl/std,cubecl/default --test spike_test spike_saxpy_runs_on_active_backend` | ✅ (verified passing) |
| PRIM-05 | SVD tall f32 matches numpy fixture (post svd_flip) | oracle | `... --test svd_test svd_tall_f32_fixture` | ❌ Wave 0 |
| PRIM-05 | SVD tall f64 matches numpy fixture (cpu only) | oracle | `--features cpu --test svd_test svd_tall_f64_fixture` | ❌ Wave 0 |
| PRIM-05 | SVD wide f32 (Aᵀ-swap) matches fixture | oracle | `... --test svd_test svd_wide_f32_fixture` | ❌ Wave 0 |
| PRIM-05 | SVD reconstruction invariant ‖UΣVᵀ−A‖ < tol | invariant | `... --test svd_test svd_reconstruction_invariant` | ❌ Wave 0 |
| PRIM-05 | SVD orthonormality ‖UᵀU−I‖ / ‖VᵀV−I‖ < tol | invariant | `... --test svd_test svd_orthonormality_invariant` | ❌ Wave 0 |
| PRIM-05 | SVD degenerate (rank-deficient / repeated / near-identity) via invariants | invariant | `... --test svd_test svd_degenerate_invariants` | ❌ Wave 0 |
| PRIM-05 | Eig symmetric f32 matches `np.linalg.eigh` (descending, reversed) | oracle | `... --test eig_test eig_symmetric_f32_fixture` | ❌ Wave 0 |
| PRIM-05 | Eig f64 (cpu) matches eigh | oracle | `--features cpu --test eig_test eig_symmetric_f64_fixture` | ❌ Wave 0 |
| PRIM-05 | Eig residual ‖A·v − λ·v‖ < tol | invariant | `... --test eig_test eig_residual_invariant` | ❌ Wave 0 |
| PRIM-05 | Eig clustered-eigenvalue case via invariant | invariant | `... --test eig_test eig_clustered_invariant` | ❌ Wave 0 |
| PRIM-05 | Moderate ~256×64 case exercises convergence loop on rocm | oracle+invariant | `... --features rocm... --test svd_test svd_moderate_256x64` | ❌ Wave 0 |
| PRIM-05 (D-11) | Bounded Jacobi scratch: allocations don't grow with sweeps | memory gate | `... --test memory_gate_test memory_gate_jacobi_scratch_bounded` | ❌ Wave 0 (extend existing) |
| PRIM-05 (D-11) | Eig reuses covariance/GEMM output buffer | memory gate | `... --test memory_gate_test memory_gate_eig_reuses_gram_buffer` | ❌ Wave 0 (extend existing) |
| PRIM-05 (D-11) | No host round-trip between sweeps (`read_backs == 1`) | memory gate | `... --test memory_gate_test memory_gate_svd_no_midsweep_readback` | ❌ Wave 0 (extend existing) |

### Sampling Rate
- **Per task commit:** `cargo test -p mlrs-backend --no-default-features --features cpu --test svd_test --test eig_test` (cpu is fast, runs f64).
- **Per wave merge:** full cpu suite + the rocm f32 suite (`--features rocm,cubecl/std,cubecl/default`).
- **Phase gate:** cpu full suite green (incl. f64) AND rocm f32 suite green AND the 3 D-11 memory-gate assertions green, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-backend/tests/svd_test.rs` — covers PRIM-05 SVD oracle + invariants (tall/wide/degenerate/moderate, f32 rocm+cpu / f64 cpu)
- [ ] `crates/mlrs-backend/tests/eig_test.rs` — covers PRIM-05 eig oracle + residual invariant
- [ ] Extend `crates/mlrs-backend/tests/memory_gate_test.rs` — 3 D-11 SVD/eig assertions
- [ ] Extend `scripts/gen_oracle.py` — `np.linalg.svd(full_matrices=False)` + `np.linalg.eigh` cases; commit `svd_*_seedNN.npz` / `eigh_*_seedNN.npz` to `tests/fixtures/`
- [ ] Apply Pattern-1 ROCm fixes (`runtime.rs` `cubecl::hip` path + `rocm` feature `cubecl/std,cubecl/default`) — prerequisite for any rocm test
- [ ] Framework install: none needed (built-in `#[test]`); numpy via `/tmp` venv only for fixture regen (PEP 668)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This is a numerical compute primitive with no network, auth, session, or user-facing input surface — most ASVS categories are N/A. The relevant surface is **unsafe device-buffer handling** and **untrusted-shape rejection**, both inherited from Phase 1/2 patterns.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library primitive) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate `(rows, cols)` against `DeviceArray::len()` and squareness (eig) **before** any `unsafe` kernel launch — return typed `PrimError` (extend the enum, e.g. `NotSquare`, `NotConverged`). Mirrors `gemm::validate_geometry`. |
| V6 Cryptography | no | — |
| V12/V14 (config/build) | partial | The `rocm` feature change touches build config; do not weaken `default-features=false` elsewhere (only add `cubecl/std`+`cubecl/default` to the `rocm` feature). |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read from a wrong caller-supplied shape | Tampering / DoS | Pre-launch geometry + squareness validation → typed `PrimError` (V5); `DeviceArray.len()` is the single source of truth for read-back size (T-04-01 mitigation, already in place) |
| Divide-by-zero / NaN propagation on rank-deficient input | DoS (poisoned result) | Near-zero floor guard in thin-U normalization (Pitfall 4); convergence cap prevents infinite sweep loop (Pitfall 5) |
| `unsafe { ArrayArg::from_raw_parts(...) }` with mismatched len | Tampering | Lengths derived from validated `DeviceArray::len()`, never from raw caller geometry — carried from Phase-2 launch idiom |

## Sources

### Primary (HIGH confidence)
- **Local build + run (this research):** `cargo build/test -p mlrs-backend --features rocm,cubecl/std,cubecl/default` — saxpy ran on gfx1100; f64 measured unsupported on rocm; f64 ran on cpu. Reproducible.
- `~/.cargo/registry/.../cubecl-0.10.0/{Cargo.toml,src/lib.rs}` — `rocm = ["hip"]`, `pub use cubecl_hip as hip`, `std` feature does NOT include `cubecl-hip`.
- `~/.cargo/registry/.../cubecl-hip-0.10.0/src/{lib.rs,runtime.rs,device.rs}` — `HipRuntime`, `AmdDevice`, `register_supported_types`.
- `~/.cargo/registry/.../cubecl-cpp-0.10.0/src/shared/base.rs:2097-2128` — `register_supported_types`; **F64 commented out (line 2115)**.
- `~/.cargo/registry/.../cubecl-common-0.10.0/build.rs` + `cubecl-runtime-0.10.0/{build.rs,src/stream/{mod.rs,event.rs}}` — `multi_threading = all(feature="std", not(wasm))` gating `MultiStream`/`ResolvedStreams`/`EventStreamBackend`.
- In-repo: `mlrs-kernels/src/reduce.rs`, `mlrs-backend/src/prims/{gemm.rs,covariance.rs}`, `pool.rs`, `device_array.rs`, `capability.rs`, `runtime.rs`, `mlrs-core/src/{sign_flip.rs,error.rs}`, `memory_gate_test.rs`, `gemm_test.rs` — patterns and gate templates.
- CubeCL manuals: `Cubecl_algebra.md`, `Cubecl_loop_control.md` (`continue` unsupported; f64 varies by GPU), `Cubecl_shared_memory.md` (single-cube tree reduction + `sync_cube`).
- `hipconfig` / `rocminfo` / `/dev/kfd` / `/dev/dri/renderD128` / `/opt/rocm/lib` — environment probes.

### Secondary (MEDIUM confidence)
- `cuml-main/cpp/include/cuml/decomposition/params.hpp` — cuSolver Jacobi defaults (`n_iterations=15`, `tol=0`) as a convergence-constant reference point (different algorithm variant; starting estimate only).

### Tertiary (LOW confidence)
- One-sided / two-sided Jacobi algorithm mechanics (rotation formulas, round-robin schedule) are standard textbook numerical linear algebra (training knowledge); the CubeCL *expression* of them is validated against in-repo patterns but not yet built — MEDIUM overall, flagged in Assumptions A1/A2.

## Metadata

**Confidence breakdown:**
- ROCm bring-up + the two fixes: HIGH — built and ran a real kernel on gfx1100 during research.
- f64-on-rocm-is-unsupported finding: HIGH — measured `f64_supported=false` + traced to `cubecl-cpp` source.
- Existing-code integration (GEMM/reduce/pool/sign reuse): HIGH — read all relevant sources.
- Jacobi kernel design: MEDIUM — algorithm is textbook-standard; CubeCL expression validated against `reduce.rs` patterns + manuals but not yet compiled. Tuning of D-12 constants + LDS layout (A1) are empirical tasks, not blockers.
- Wide-path tolerance: MEDIUM — exact algebra, pending one oracle run (Open Q4).

**Research date:** 2026-06-12
**Valid until:** 2026-07-12 (30 days; stable — cubecl 0.10 pin is fixed, ROCm env is fixed). Re-verify if `cubecl` is bumped (the F64-registration and `multi_threading` gating could change in a later version, potentially enabling f64 on rocm).

## RESEARCH COMPLETE
