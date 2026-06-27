---
phase: 17
slug: randomforest-gpu-histogram-split-feasibility-spike-gating
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-27
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `17-RESEARCH.md` § Validation Architecture. Refined by the planner/executor.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust) + spike live-launch harness (`crates/mlrs-backend/tests/spike_test.rs` shape) |
| **Config file** | none — uses existing workspace `Cargo.toml` + cargo features `cpu`, `rocm` |
| **Quick run command** | `cargo test -p mlrs-backend --features cpu <targeted_test> -- --nocapture` |
| **Full suite command** | `cargo test -p mlrs-backend --features cpu` (gate: cpu f64) + `cargo test -p mlrs-backend --features rocm` (gate: rocm f32) |
| **Estimated runtime** | targeted spike tests seconds–minutes; full mlrs-backend cpu suite ~6 min (run targeted post-merge gates) |

---

## Sampling Rate

- **After every task commit:** Run the targeted spike/witness test for the touched kernel
- **After every plan wave:** Run the cpu(f64) gate for the affected crate
- **Before `/gsd-verify-work`:** cpu(f64) gate green + rocm(f32) gate green-or-skip-with-log; VERDICT.md present
- **Max feedback latency:** keep targeted runs under a few minutes; background the full suite (see memory: backend test suite slow)

---

## Per-Task Verification Map

> Filled by the planner once PLAN tasks exist. Every kernel task carries a VALUE-asserting
> live-launch test (never non-panic) — the 002-B silent-miscompile backstop.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | TREE-01 | — | N/A (local compute spike) | unit/live-launch | `cargo test -p mlrs-backend --features cpu <hist_test> -- --nocapture` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Tier-1 sklearn oracle fixture(s) for `DecisionTreeClassifier(gini)` + `DecisionTreeRegressor(squared_error)` on injected fixed bootstrap/feature indices — generated via numpy venv (committed blob; gen rule lives in the generator, never hand-patched — Phase 13 CR-01/CR-02 lesson)
- [ ] Adversarial/tie fixture for the seed-from-first argmax (silent-miscompile backstop)
- [ ] Live-launch harness module(s) cloned from `crates/mlrs-backend/tests/spike_test.rs`

*Existing cargo + spike_test.rs infrastructure otherwise covers the phase.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| A1–A5 abort-signal evaluation + GO/ADJUST/ABORT verdict | TREE-01 | Judgment synthesis over benchmark + correctness evidence | Author reviews benchmark numbers + witness results against the A1–A5 table in RESEARCH.md and records the verdict in VERDICT.md |
| rocm(f32) gate | TREE-01 | Requires gfx1100/ROCm hardware; f64 unsupported on rocm | Run rocm-feature tests on GPU host; f64 paths skip-with-log |

---

## Validation Sign-Off

- [ ] All kernel tasks have a VALUE-asserting live-launch verify or Wave 0 fixture dependency
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers the oracle fixtures + adversarial/tie fixture
- [ ] No watch-mode flags
- [ ] Feedback latency acceptable (targeted runs; full suite backgrounded)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
