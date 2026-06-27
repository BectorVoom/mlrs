---
phase: 17
slug: randomforest-gpu-histogram-split-feasibility-spike-gating
status: approved
nyquist_compliant: true
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

> Synced to the finalized plans (17-01..17-05). Every kernel/witness task carries a VALUE-asserting
> live-launch test (never non-panic) — the 002-B silent-miscompile backstop. Threat refs from each
> plan's `<threat_model>`; this is a local offline compute spike (no high web-app threats).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 17-01-01 | 01 | 1 | TREE-01 | — | N/A (dev-controlled fixtures) | unit (python) | `python3 -c "import ast; ast.parse(open('scripts/gen_oracle.py').read())"` + grep `gen_decision_tree_clf`/`gen_decision_tree_reg` (≥2) | ❌ W0 | ⬜ pending |
| 17-01-02 | 01 | 1 | TREE-01 | non-circular oracle | gen rule in generator, not hand-patched blob | unit (python) | regen → `ls tests/fixtures/ \| grep -E 'tree_dt_(clf\|reg)(_adv)?_(f32\|f64)_seed42\.npz'` (≥6) | ❌ W0 | ⬜ pending |
| 17-02-01 | 02 | 1 | TREE-01 | 002-A / banned-set | no SharedMemory/Atomic/F::INFINITY/mutable-bool | build | `cargo build -p mlrs-backend --features cpu --tests` + grep 3 kernel fns | ❌ W0 | ⬜ pending |
| 17-02-02 | 02 | 1 | TREE-01 | 002-A all-zeros | VALUE read-back ≠ zeros, per kernel | live-launch | `cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture` | ❌ W0 | ⬜ pending |
| 17-03-01 | 03 | 2 | TREE-01 | A5 correctness | exact structure + ≤1e-5 leaf vs sklearn | live-launch | `cargo test -p mlrs-backend --features cpu --test tree_witness -- --nocapture` | ❌ W0 | ⬜ pending |
| 17-03-02 | 03 | 2 | TREE-01 | 002-B silent miscompile | adversarial tie + pure-leaf VALUE-assert | live-launch | `cargo test -p mlrs-backend --features cpu --test tree_witness adversarial -- --nocapture` | ❌ W0 | ⬜ pending |
| 17-04-01 | 04 | 2 | TREE-01 | A3 cost | benchmark prints 64/128-bin wall-clock + sweep | bench (Instant) | `cargo test -p mlrs-backend --features cpu --test tree_bench -- --nocapture` | ❌ W0 | ⬜ pending |
| 17-05-01 | 05 | 3 | TREE-01 | T-17-09 decision integrity | A1–A5 evidence-cited verdict | file-assert | grep `A[1-5]`(≥5) + `GO\|ADJUST\|ABORT` + `colid` + `tier` in VERDICT.md | ❌ W0 | ⬜ pending |
| 17-05-02 | 05 | 3 | TREE-01 | T-17-10 evidence tamper | verbatim copy, live tests authoritative | file-assert | spike dirs 003–006 exist + MANIFEST rows (≥4) | ❌ W0 | ⬜ pending |
| 17-05-03 | 05 | 3 | TREE-01 | T-17-09 (human gate) | blocking human confirmation of verdict | manual (checkpoint) | human verify per Plan 05 Task 3 `<how-to-verify>` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · File Exists ❌ W0 = produced during this phase's waves*

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

- [x] All kernel tasks have a VALUE-asserting live-launch verify or Wave 0 fixture dependency
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (Wave 1: 4/4 auto; Wave 2: 3/3 auto; Wave 3: 2 auto + 1 checkpoint)
- [x] Wave 0 (Plan 01) covers the oracle fixtures + adversarial/tie fixture
- [x] No watch-mode flags (all commands are targeted `--test <file>`; bench uses Instant, not Criterion)
- [x] Feedback latency acceptable (targeted runs; full suite backgrounded)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-27
