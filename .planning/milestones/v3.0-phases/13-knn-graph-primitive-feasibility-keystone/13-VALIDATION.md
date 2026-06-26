---
phase: 13
slug: knn-graph-primitive-feasibility-keystone
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-23
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (oracle-fixture comparison vs sklearn) |
| **Config file** | none — Wave 0 adds `knn_graph_test.rs` + per-metric fixtures |
| **Quick run command** | `cargo test -p mlrs-backend --features cpu knn_graph` |
| **Full suite command** | `cargo test -p mlrs-backend --features cpu` |
| **Estimated runtime** | ~{N} seconds (mlrs-backend cpu suite is slow — prefer targeted run) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p mlrs-backend --features cpu knn_graph`
- **After every plan wave:** Run targeted KNN + distance tests (full suite backgrounded)
- **Before `/gsd-verify-work`:** Full targeted KNN suite must be green (per metric)
- **Max feedback latency:** {N} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | PRIM-11 | — / — | N/A | unit | `cargo test -p mlrs-backend --features cpu knn_graph` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `knn_graph_test.rs` — oracle-comparison stubs for PRIM-11 (per metric; must include a duplicate-point row asserting VALUES, not just shape)
- [ ] per-metric `gen_oracle.py` fixtures (Euclidean, Manhattan, Cosine, Chebyshev, Minkowski-p) incl. duplicate-point design
- [ ] new KNN-graph kernel module + `prims/mod.rs` registration

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm f32 launch | PRIM-11 | rocm GPU gate verified opportunistically | Run targeted KNN test with `--features rocm` on gfx1100 |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
