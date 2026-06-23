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

### Calibrated property-gate thresholds (Plan 04 — Spike flag item 2)

Measured on the first fixture run (cpu-MLIR f64, `n=60`, 3 well-separated
clusters, `random_state=42`, `n_epochs=200`) by computing mlrs's and
umap-learn 0.5.12's structural scores on identical seeded data. The gate is
RELATIVE to umap (`mlrs ≥ umap − ε`, D-04), never an absolute floor.

| Metric | trustworthiness (mlrs / umap) | margin (umap−mlrs) | kNN-overlap (mlrs / umap) | margin (umap−mlrs) | downstream-ARI (mlrs / umap) | gap |
|--------|-------------------------------|--------------------|----------------------------|--------------------|------------------------------|-----|
| euclidean | 0.9673 / 0.9680 | **+0.0007** | 0.6950 / 0.6917 | −0.0033 | 1.0000 / 1.0000 | 0.0000 |
| manhattan | 0.9654 / 0.9635 | −0.0019 | 0.6850 / 0.6783 | −0.0067 | 1.0000 / 1.0000 | 0.0000 |
| cosine | 0.9685 / 0.9673 | −0.0012 | 0.6983 / 0.6733 | −0.0250 | 1.0000 / 1.0000 | 0.0000 |
| chebyshev | 0.9621 / 0.9615 | −0.0006 | 0.6350 / 0.6317 | −0.0033 | 1.0000 / 1.0000 | 0.0000 |
| minkowski (p=3) | 0.9675 / 0.9652 | −0.0023 | 0.6917 / 0.6667 | −0.0250 | 1.0000 / 1.0000 | 0.0000 |

**Worst positive margin:** trust +0.0007 (euclidean); overlap +0.0000 (mlrs ≥ umap on every metric); ARI gap 0.0000 on all five.

**Calibrated constants** (in `crates/mlrs-algos/tests/umap_test.rs`):

| Constant | Value | Rationale |
|----------|-------|-----------|
| `PROPERTY_EPS` (trust + overlap slack) | **0.02** | ≈28× the worst trust margin (0.0007); tight while absorbing cpu/rocm structural jitter (D-04). |
| `ARI_BAND` (downstream-ARI band) | **0.05** | ARI gap measured 0.0000 on all metrics; a tight relative band, not an absolute floor. |

The downstream-ARI clusters BOTH embeddings with the same deterministic host
Lloyd k-means (`k = 3` true classes) and scores each clustering against the
true labels — both mlrs and umap recover the 3 clusters exactly. All 5
`layout_property_<metric>` tests are GREEN at these thresholds; the full run
(spectral-init Jacobi eig dominated) took ~1717s for the five metrics.

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
