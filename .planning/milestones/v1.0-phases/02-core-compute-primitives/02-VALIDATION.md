---
phase: 02
slug: core-compute-primitives
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-12
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (cargo test); integration tests in `crates/*/tests/` |
| **Config file** | none — standard cargo layout (Phase 1 convention) |
| **Quick run command** | `cargo test -p mlrs-backend --features cpu <test_name>` |
| **Full suite command** | `cargo test -p mlrs-backend --features cpu && cargo test -p mlrs-backend --features wgpu` |
| **Estimated runtime** | ~60 seconds (cpu); wgpu adds adapter-dependent time |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p mlrs-backend --features cpu <prim>_test`
- **After every plan wave:** Run `cargo test -p mlrs-backend --features cpu && cargo test -p mlrs-backend --features wgpu`
- **Before `/gsd-verify-work`:** Full suite must be green on cpu AND wgpu (f32 always, f64 where `skip_f64_with_log` permits)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-XX | 01 | 1 | PRIM-01 | — | N/A (numeric kernel; `rows*cols==len` shape assert) | integration | `cargo test -p mlrs-backend --features cpu gemm` | ❌ W0 | ⬜ pending |
| 02-02-XX | 02 | 2 | PRIM-02 | — | N/A | integration | `cargo test -p mlrs-backend --features wgpu reduce` | ❌ W0 | ⬜ pending |
| 02-03-XX | 03 | 3 | PRIM-03 | — | N/A | integration | `cargo test -p mlrs-backend --features cpu distance` | ❌ W0 | ⬜ pending |
| 02-04-XX | 04 | 3 | PRIM-04 | — | N/A | integration | `cargo test -p mlrs-backend --features cpu covariance` | ❌ W0 | ⬜ pending |
| 02-05-XX | 05 | 4 | D-10 memory gate | — | N/A | integration | `cargo test -p mlrs-backend --features cpu memory_gate` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Task IDs finalized by the planner.*

---

## Wave 0 Requirements

- [ ] `tests/gemm_test.rs`, `tests/reduce_test.rs`, `tests/distance_test.rs`, `tests/covariance_test.rs`, `tests/memory_gate_test.rs` — stubs covering PRIM-01..04 + D-10
- [ ] `PoolStats.read_backs` counter (or read-back-instrumented `to_host`) — enables D-10 gate 2 as a runtime assertion
- [ ] Subgroup capability-query probe (mirror `spike_capability_query_reports_f64`) for the plane-path skip gate
- [ ] D-12 convention `.npz` fixtures (GEMM, distance squared/sqrt, cov ddof=0/1) via `gen_oracle.py` (/tmp venv, PEP 668)
- [ ] **GEMM substrate DECISION task** (Open Question 1) before any GEMM code — `checkpoint:human-verify`
- [ ] Framework install: none — cargo test already in use.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GEMM substrate decision (wrap cubecl-matmul vs hand-write) | PRIM-01 | Requires human confirmation of whether a cubecl-0.10-compatible matmul source exists | `checkpoint:human-verify` decision task in GEMM Wave 0; default = hand-write tiled GEMM |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
