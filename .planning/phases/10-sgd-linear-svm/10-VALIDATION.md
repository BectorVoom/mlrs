---
phase: 10
slug: sgd-linear-svm
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust) + pinned-deterministic sklearn oracle fixtures (committed blobs) |
| **Config file** | none — workspace `Cargo.toml`; oracle gen via `gen_oracle.py` (/tmp venv, numpy+sklearn) |
| **Quick run command** | `cargo test -p mlrs-prims --features cpu sgd` |
| **Full suite command** | `cargo test --features cpu` (targeted per [[backend-test-suite-slow]] / [[full-cargo-test-exhausts-disk]]) |
| **Estimated runtime** | ~targeted (full cpu suite ~6min; run targeted post-merge gates) |

---

## Sampling Rate

- **After every task commit:** Run the targeted prim/estimator test (e.g. `cargo test -p mlrs-prims --features cpu sgd`)
- **After every plan wave:** Run the affected crate's `--features cpu` tests (targeted, not the full suite)
- **Before `/gsd-verify-work`:** Targeted cpu (f64) + rocm (f32) gates must be green per [[rocm-is-runnable-gpu-gate]]
- **Max feedback latency:** targeted tests; background the full suite

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 0 | PRIM-10 | — | N/A | unit | `cargo test -p mlrs-prims --features cpu sgd` | ❌ W0 | ⬜ pending |

*Populated by the planner / Nyquist scaffold pass. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] SGD-prim standalone convex-objective test (`--features cpu` launch, not just compile) — PRIM-10
- [ ] Pinned-deterministic sklearn oracle fixtures (`shuffle=False`, fixed `eta0`/schedule, fixed `max_iter`, `tol=0`) for the four estimators — SGDSVM-01…04
- [ ] PoolStats memory gate per prim

*Filled in detail by the planner's Wave-0 scaffold per the 07-01/08-01/09-01 precedent.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| f32-on-rocm weight band | recurring gate | rocm runs f32 only ([[rocm-is-runnable-gpu-gate]]); documented band, not exact-equal | Run rocm f32 oracle; confirm weights within documented band |

*Exact predicted labels (classifiers) are the HARD automated gate. f64 oracle cases use `skip_f64_with_log`.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable (targeted tests)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
