---
phase: 8
slug: kernel-family
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 08-RESEARCH.md "## Validation Architecture". Oracle = scikit-learn
> (forced exact for KernelDensity). Gate = cpu(f64) + rocm(f32); every f64 oracle
> case behind `skip_f64_with_log`.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (prim tests in `crates/mlrs-backend/tests/`; algo tests in `crates/mlrs-algos/tests/`; py via existing harness) |
| **Config file** | none (cargo); oracle fixtures `tests/fixtures/*.npz` via `scripts/gen_oracle.py` ([[oracle-fixture-regen-needs-venv]]) |
| **Quick run command** | `cargo test --features cpu -p mlrs-backend kernel_matrix` (targeted — suite is slow, [[backend-test-suite-slow]]) |
| **Full suite command** | `cargo test --features cpu` then opportunistic `cargo test --features rocm` (f32 only) |
| **Estimated runtime** | targeted ~5–30s; full cpu suite ~6 min (background it) |

---

## Sampling Rate

- **After every task commit:** Run targeted `cargo test --features cpu <new_test>` (avoid the full slow suite)
- **After every plan wave:** Run `cargo test --features cpu -p mlrs-backend` (prim) then `-p mlrs-algos` (estimators)
- **Before `/gsd-verify-work`:** Full `cargo test --features cpu` green + opportunistic `--features rocm` (f32); every f64 case behind `skip_f64_with_log`
- **Max feedback latency:** ~30 seconds (targeted), accepted ~6 min for full-suite wave gates

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 8-01-xx | 01 | 0 | PRIM-08 / KERNEL-01 / KERNEL-02 | — | N/A (numeric kernel, no untrusted input) | scaffold | `cargo test -p mlrs-algos -- --ignored score_samples` | ❌ W0 | ⬜ pending |
| 8-02-xx | 02 | 1 | PRIM-08 | — | N/A | unit (prim) | `cargo test --features cpu -p mlrs-backend kernel_matrix` | ❌ W0 | ⬜ pending |
| 8-02-xx | 02 | 1 | PRIM-08 | — | bounded device memory | unit (prim) | `cargo test --features cpu -p mlrs-backend kernel_matrix_memory_gate` | ❌ W0 | ⬜ pending |
| 8-03-xx | 03 | 2 | KERNEL-01 | — | N/A | integration | `cargo test --features cpu -p mlrs-algos kernel_ridge` | ❌ W0 | ⬜ pending |
| 8-04-xx | 04 | 2 | KERNEL-02 | — | N/A | integration | `cargo test --features cpu -p mlrs-algos kernel_density` | ❌ W0 | ⬜ pending |
| 8-05-xx | 05 | 3 | PY-06 (share) | — | dtype dispatch / `guard_f64()` | py/smoke | existing py test harness | ❌ W0 | ⬜ pending |

*Wave/plan IDs are indicative — the planner sets final wave assignment. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-backend/tests/kernel_matrix_test.rs` — PRIM-08 values (linear/rbf/poly/sigmoid, f32+f64) + PoolStats memory gate (mirror `incremental_svd_test.rs` gate shape)
- [ ] `crates/mlrs-algos/tests/kernel_ridge_test.rs` — KERNEL-01 (mirror `ridge_test.rs`): predict ≤ 1e-5, 4 kernels, multi-target, gamma None + explicit
- [ ] `crates/mlrs-algos/tests/kernel_density_test.rs` — KERNEL-02 (new): score_samples ≤ documented tol, 6 kernels, scott/silverman, ScoreSamples shape
- [ ] `scripts/gen_oracle.py` extensions: `gen_kernel_matrix`, `gen_kernel_ridge`, `gen_kernel_density` (numpy/sklearn, committed `.npz`; KernelDensity oracle forced exact `rtol=0, atol=0`)
- [ ] `#[ignore]` scaffold tests in a Wave-0 plan (mirror Phase-7 07-01 scaffold: trait + AlgoError guards + module index + ignored tests + oracle generators)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm f32 numerical bands (KernelRidge predictions, KernelDensity large-dynamic-range log-density) | KERNEL-01 / KERNEL-02 | ROCm gate is opportunistic (gfx1100); f64 unsupported on rocm | Run `cargo test --features rocm kernel_*` on a gfx1100 host; assert within the documented per-family f32 band, not ≤ 1e-5 ([[rocm-is-runnable-gpu-gate]]) |

*All other phase behaviors have automated cpu(f64) verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (kernel_matrix / kernel_ridge / kernel_density test files + oracle generators)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (targeted)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
