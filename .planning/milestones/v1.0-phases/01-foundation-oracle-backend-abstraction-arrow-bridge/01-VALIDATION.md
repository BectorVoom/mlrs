---
phase: 1
slug: foundation-oracle-backend-abstraction-arrow-bridge
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust built-in; tests in `tests/` or `*_test.rs` per AGENTS.md) |
| **Config file** | none — Wave 0 creates the workspace + per-crate test dirs |
| **Quick run command** | `cargo test --features cpu` |
| **Full suite command** | `cargo test --features cpu && cargo test --features wgpu` |
| **Estimated runtime** | ~{N} seconds (TBD after Wave 0) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --features cpu`
- **After every plan wave:** Run `cargo test --features cpu && cargo test --features wgpu`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** {N} seconds (TBD)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 0 | FOUND-{XX} | — | N/A | unit | `cargo test --features cpu` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*(Populated by the planner / Nyquist auditor once PLAN.md tasks exist.)*

---

## Wave 0 Requirements

- [ ] Cargo workspace + five crates scaffold — required before any test can compile
- [ ] Toolchain/API spike (hello-world `#[cube]` kernel + capability query + npz load on cpu and wgpu) — resolves RESEARCH assumptions A1–A7
- [ ] Oracle fixtures committed (`.npz`) + `scripts/gen_oracle.py`

*(RESEARCH.md enumerates the Wave-0 gaps; expand per PLAN.md.)*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `--features cuda` compiles (without running) | FOUND-01 | No CUDA device in CI/this environment | `cargo build --features cuda` succeeds; run not required |
| f64 skip/xfail on wgpu adapters lacking `SHADER_F64` | FOUND-07 | Depends on the runtime adapter's capabilities | Inspect CI log for logged skip reason + dtype/backend line |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
