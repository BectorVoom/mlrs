---
phase: 14
slug: umap
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-23
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `cargo test` (oracle-fixture value-gates + property/structural gates) |
| **Config file** | none — tests live in `crates/mlrs-algos/tests/umap_test.rs` (tests separated from source, AGENTS.md §2) |
| **Quick run command** | `cargo test -p mlrs-algos --features cpu umap` |
| **Full suite command** | `cargo test -p mlrs-algos --features cpu umap && cargo test -p mlrs-algos --features rocm umap` (f64 cpu + f32 rocm gate; f64-on-rocm skips-with-log) |
| **Estimated runtime** | ~{N} seconds (targeted UMAP tests; full backend suite is slow — keep gates targeted) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p mlrs-algos --features cpu umap`
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** {N} seconds

---

## Per-Task Verification Map

> Derived during planning. Deterministic stages (KNN graph, smooth-kNN ρ/σ, fuzzy union, spectral init, a/b) value-gate ≤1e-5 f64 against committed umap-learn 0.5.12 oracle fixtures × all 5 metrics; the stochastic SGD layout property-gates (trustworthiness / kNN-overlap ≥ umap-learn − ε, downstream-ARI within band) + byte-identical-across-runs per (backend,dtype).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | UMAP-01 | — | N/A | value-gate | `cargo test -p mlrs-algos --features cpu umap` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-algos/tests/umap_test.rs` — oracle-fixture value-gate + property-gate test scaffolding for UMAP-01..04
- [ ] umap-learn 0.5.12 oracle fixtures (committed blobs) regenerated via the `/tmp` numpy/umap-learn venv — all 5 metrics; intermediates: graph rows/cols/vals, sigmas/rhos, a/b, spectral-init coords
- [ ] property-gate helpers (trustworthiness, kNN-overlap, downstream-ARI) in-repo

*Calibrated thresholds (ε, ARI band) recorded here after the first oracle-fixture/spike run (ROADMAP Spike flag item 2).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Python/PyO3 estimator surface | (deferred Phase 16) | no maturin/pyarrow in this env — routes to UAT | N/A this phase |

*All in-scope Phase-14 behaviors have automated Rust verification (value-gate + property-gate).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
