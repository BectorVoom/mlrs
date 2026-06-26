---
phase: 15
slug: hdbscan
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-24
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `15-RESEARCH.md` § Validation Architecture + § Security Domain.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (`cargo test`), per-backend via `--features cpu` / `--features rocm` |
| **Config file** | none — workspace `cargo test`; fixtures under `crates/mlrs-algos/tests/fixtures/*.npz` |
| **Quick run command** | `cargo test --features cpu --test hdbscan_test <targeted_fn> -- --nocapture` |
| **Full suite command** | `cargo test --features cpu --test hdbscan_test` (targeted file only) |
| **Estimated runtime** | targeted fn ~seconds; whole `hdbscan_test` file < ~1min. AVOID full backend `cargo test --features cpu` (~6min, exhausts disk — project memory). |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --features cpu --test hdbscan_test <targeted_fn> -- --nocapture` (one metric/score at a time).
- **After every plan wave:** Run `cargo test --features cpu --test hdbscan_test` (whole HDBSCAN file).
- **Before `/gsd-verify-work`:** `hdbscan_test` green on cpu(f64); rocm(f32) spot-check; f64-on-rocm skips-with-log confirmed.
- **Max feedback latency:** ~60 seconds (targeted file).

---

## Per-Task Verification Map

| Requirement | Behavior | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-------------|----------|------------|-----------------|-----------|-------------------|-------------|--------|
| HDBS-02 | labels exact-up-to-perm (`-1` pinned), per metric × {euclidean,l1,cosine,chebyshev,minkowski,precomputed} × {f32,f64} | T-15-V5 | precomputed X square+symmetric validated before back-end | unit/oracle | `cargo test --features cpu --test hdbscan_test labels_match_sklearn` | ❌ W0 | ⬜ pending |
| HDBS-02 | tie-heavy + duplicate-point fixture exactness (D-04 TRUE GATE) | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test tie_break_exact` | ❌ W0 | ⬜ pending |
| HDBS-01 | `probabilities_` ≤1e-5 vs sklearn (D-06) | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test probabilities_match` | ❌ W0 | ⬜ pending |
| HDBS-03 | GLOSH `outlier_scores_` ≤1e-5 vs hdbscan 0.8.44 | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test outlier_scores_match` | ❌ W0 | ⬜ pending |
| HDBS-04 | `centroids_`/`medoids_` ≤1e-5 vs sklearn (same perm) | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test centers_match` | ❌ W0 | ⬜ pending |
| HDBS-01/D-09 | non-default eom/leaf, ε>0, max_cluster_size, alpha — exact labels | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test selection_knobs` | ❌ W0 | ⬜ pending |
| HDBS-01/D-09 | build validation: `min_samples>=1` (Some), `max_cluster_size` 0 or `>=min_cluster_size`, `alpha>0`, `minkowski p>=1` | T-15-V5 | typed `BuildError`/`AlgoError` before any `unsafe` launch | unit | `cargo test --features cpu --test hdbscan_test build_validation` | ❌ W0 | ⬜ pending |
| HDBS-01 | memory: device front-end no n×n leak (PoolStats gate) | T-15-OVF | `checked_mul` before `pool.acquire`; query-axis-tiled | unit | `cargo test --features cpu --test hdbscan_test memory_gate` | ❌ W0 | ⬜ pending |
| HDBS-01 | edge cases: all-noise, single cluster, n<mcs (match sklearn) | — | N/A | unit/oracle | `cargo test --features cpu --test hdbscan_test edge_cases` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-algos/tests/hdbscan_test.rs` — REPLACE the 4 shell convention tests with the oracle gates above (the shell all-`-1` `fit_roundtrip` test is removed — `fit` no longer returns all-`-1`).
- [ ] `scripts/gen_oracle.py` — add `gen_hdbscan_*` generators (sklearn + hdbscan 0.8.44; blob, nested-density, tie-heavy, duplicate-point designs; per metric; f32+f64).
- [ ] `crates/mlrs-algos/tests/fixtures/hdbscan_*_seed*.npz` — committed blobs (regen via `/tmp` venv with numpy + `hdbscan==0.8.44`, PEP 668).
- [ ] `crates/mlrs-core/src/label_perm.rs` — `-1→-1`-pinned matcher + its own unit test.
- [ ] hdbscan library install: `/tmp/<venv>/bin/pip install hdbscan==0.8.44` (fixture-gen host only — NOT a runtime/test dep).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm(f32) backend gate | HDBS-01/02 | rocm device not in CI; gfx1100/ROCm 7.1.1 opportunistic | Run `cargo test --features rocm --test hdbscan_test` on a gfx1100 host; confirm f32 green and f64-on-rocm skips-with-log. |

*All other phase behaviors have automated cpu-gate verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (fixtures, generators, label_perm)
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
