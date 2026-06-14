---
phase: 3
slug: svd-eigendecomposition-primitive-hard-gate
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-12
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `03-RESEARCH.md` §Validation Architecture. PRIM-05.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` — integration tests in `tests/` (AGENTS.md §2: no in-source `mod tests`) |
| **Config file** | none — `cargo test` with per-feature backend selection |
| **Quick run command** | `cargo test -p mlrs-backend --no-default-features --features cpu --test svd_test --test eig_test` |
| **Full suite command** | `cargo test -p mlrs-backend --no-default-features --features cpu` (f64) + `cargo test -p mlrs-backend --no-default-features --features rocm,cubecl/std,cubecl/default` (f32 GPU gate) |
| **Estimated runtime** | ~30–90 seconds (cpu fast; rocm includes first HIP compile) |

> **Backend split (RESEARCH correction to D-07):** f64 validates on **cpu** (rocm/cubecl-cpp has F64 unsupported at the library layer); f32 validates on **rocm** (gfx1100). The `skip_f64_with_log` gate already encodes this. The rocm invocation requires the Pattern-1 Cargo fix (`rocm` feature adds `cubecl/std`+`cubecl/default`); after that fix, plain `--features rocm` works.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p mlrs-backend --no-default-features --features cpu --test svd_test --test eig_test` (cpu is fast, exercises f64).
- **After every plan wave:** Run full cpu suite **+** the rocm f32 suite (`--features rocm,cubecl/std,cubecl/default`).
- **Before `/gsd-verify-work`:** cpu full suite green (incl. f64) **AND** rocm f32 suite green **AND** the 3 D-11 memory-gate assertions green.
- **Max feedback latency:** ~90 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | PRIM-05 | — | ROCm bring-up: real `#[cube]` kernel runs on gfx1100 + correct read-back | smoke | `cargo test -p mlrs-backend --no-default-features --features rocm,cubecl/std,cubecl/default --test spike_test spike_saxpy_runs_on_active_backend` | ✅ verified passing in research | ⬜ pending |
| TBD | — | — | PRIM-05 | T-V5 | shape/squareness validated pre-launch → typed `PrimError` | oracle | `--features cpu --test svd_test svd_tall_f32_fixture` (also rocm f32) | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | f64 path runs (cpu) | oracle | `--features cpu --test svd_test svd_tall_f64_fixture` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | wide Aᵀ-swap holds tol | oracle | `--test svd_test svd_wide_f32_fixture` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | reconstruction ‖UΣVᵀ−A‖ < tol | invariant | `--test svd_test svd_reconstruction_invariant` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | orthonormality ‖UᵀU−I‖/‖VᵀV−I‖ < tol | invariant | `--test svd_test svd_orthonormality_invariant` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | T-DoS | degenerate (rank-deficient/repeated/near-identity) via invariants; near-zero floor + convergence cap | invariant | `--test svd_test svd_degenerate_invariants` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | eig matches `np.linalg.eigh` (descending, reversed) | oracle | `--test eig_test eig_symmetric_f32_fixture` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | eig f64 (cpu) matches eigh | oracle | `--features cpu --test eig_test eig_symmetric_f64_fixture` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | eig residual ‖A·v − λ·v‖ < tol | invariant | `--test eig_test eig_residual_invariant` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | clustered-eigenvalue case via invariant | invariant | `--test eig_test eig_clustered_invariant` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 | — | moderate ~256×64 exercises convergence loop on rocm | oracle+invariant | `--features rocm,cubecl/std,cubecl/default --test svd_test svd_moderate_256x64` | ❌ W0 | ⬜ pending |
| TBD | — | — | PRIM-05 (D-11) | — | bounded Jacobi scratch: allocations don't grow with sweeps | memory gate | `--test memory_gate_test memory_gate_jacobi_scratch_bounded` | ❌ W0 (extend) | ⬜ pending |
| TBD | — | — | PRIM-05 (D-11) | — | eig reuses covariance/GEMM output buffer | memory gate | `--test memory_gate_test memory_gate_eig_reuses_gram_buffer` | ❌ W0 (extend) | ⬜ pending |
| TBD | — | — | PRIM-05 (D-11) | — | no host round-trip between sweeps (`read_backs == 1`) | memory gate | `--test memory_gate_test memory_gate_svd_no_midsweep_readback` | ❌ W0 (extend) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*Task IDs are assigned by the planner; this map binds PRIM-05 behaviors to automated commands.*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-backend/tests/svd_test.rs` — PRIM-05 SVD oracle + invariants (tall/wide/degenerate/moderate; f32 rocm+cpu, f64 cpu)
- [ ] `crates/mlrs-backend/tests/eig_test.rs` — PRIM-05 eig oracle + residual invariant + clustered case
- [ ] Extend `crates/mlrs-backend/tests/memory_gate_test.rs` — 3 D-11 SVD/eig assertions
- [ ] Extend `scripts/gen_oracle.py` — `np.linalg.svd(full_matrices=False)` + `np.linalg.eigh` cases; commit `svd_*_seedNN.npz` / `eigh_*_seedNN.npz` to `tests/fixtures/` (numpy via `/tmp` venv, PEP 668; fixtures are committed blobs)
- [ ] Apply Pattern-1 ROCm fixes (`runtime.rs` `cubecl::hip` import path + `rocm` feature `cubecl/std,cubecl/default` in Cargo.toml) — prerequisite for any rocm test
- [ ] Framework install: none needed (built-in `#[test]`)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Oracle fixture regeneration | PRIM-05 | Needs a `/tmp` numpy venv (PEP 668 externally-managed); fixtures are committed blobs, not regenerated at test time | `python3 -m venv /tmp/oraclevenv && /tmp/oraclevenv/bin/pip install numpy && /tmp/oraclevenv/bin/python scripts/gen_oracle.py` |

*All runtime phase behaviors otherwise have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (svd_test.rs, eig_test.rs, memory_gate extensions, gen_oracle.py cases, ROCm Cargo fix)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
