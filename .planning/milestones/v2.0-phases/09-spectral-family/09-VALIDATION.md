---
phase: 9
slug: spectral-family
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `09-RESEARCH.md` → Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` — integration tests under `crates/*/tests/` only (AGENTS.md §2 — never in-source `mod tests`) |
| **Config file** | none (cargo standard) |
| **Quick run command** | `cargo test --features cpu -p mlrs-backend laplacian_test` (targeted) |
| **Full suite command** | `cargo test --features cpu` (slow ~6min + disk pressure — prefer targeted gates, background the full run) |
| **Estimated runtime** | ~30s targeted oracle case; full suite ~6min |

*Oracle harness:* committed `.npz` blobs via `mlrs_core::oracle::load_npz`; regen `scripts/gen_oracle.py` in a `/tmp` venv (numpy+scipy+sklearn, PEP 668).

---

## Sampling Rate

- **After every task commit:** Run the targeted `cargo test --features cpu -p <crate> <test_name>` (sub-30s for laplacian/spectral oracle cases)
- **After every plan wave:** The phase's own test files green on cpu (f32+f64); rocm f32 opportunistic
- **Before `/gsd-verify-work`:** All Phase-9 tests green on cpu(f64)+rocm(f32); f64-on-rocm skips-with-log
- **Max feedback latency:** ~30 seconds (targeted)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 9-PRIM-01 | laplacian | 1 | PRIM-09 | — | `L = I − D^-1/2 A D^-1/2` vs host reference, f32+f64 | unit (oracle) | `cargo test --features cpu -p mlrs-backend laplacian_test` | ❌ W0 | ⬜ pending |
| 9-PRIM-02 | laplacian | 1 | PRIM-09 | T-9-LAP | No NaN/inf on zero-degree node (isolated-node fixture); typed-zero guard | unit | `cargo test --features cpu -p mlrs-backend laplacian_test::zero_degree` | ❌ W0 | ⬜ pending |
| 9-PRIM-03 | laplacian | 1 | PRIM-09 | T-9-LDS | PoolStats memory gate (reuse bounded, no mid-pipeline readback) | unit | `cargo test --features cpu -p mlrs-backend laplacian_test::memory_gate` | ❌ W0 | ⬜ pending |
| 9-SE-01 | spectral-embedding | 2 | SPECTRAL-01 | — | `embedding_` value-match (rbf) after sign align, f64 strict / f32 band | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_embedding_test` | ❌ W0 | ⬜ pending |
| 9-SE-02 | spectral-embedding | 2 | SPECTRAL-01 | — | `embedding_` value-match (nearest_neighbors default) | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::knn_affinity` | ❌ W0 | ⬜ pending |
| 9-SE-03 | spectral-embedding | 2 | SPECTRAL-01 | — | degenerate-spectrum subspace test (D-09) | unit | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::subspace` | ❌ W0 | ⬜ pending |
| 9-SE-04 | spectral-embedding | 2 | SPECTRAL-01 | T-9-VAL | `n_samples > 64` → typed `AlgoError` BEFORE device (D-06) | unit | `cargo test --features cpu -p mlrs-algos spectral_embedding_test::reject_oversize` | ❌ W0 | ⬜ pending |
| 9-SC-01 | spectral-clustering | 3 | SPECTRAL-02 | — | `labels_` match up to permutation, well-separated fixture (D-10) | unit (oracle) | `cargo test --features cpu -p mlrs-algos spectral_clustering_test` | ❌ W0 | ⬜ pending |
| 9-PY-01 | py-bindings | 3 | PY-06 (share) | T-9-VAL | PyO3 smoke: fit + `embedding_`/`labels_` accessors, f32+f64 | smoke | `cargo test -p mlrs-py spectral` (or maturin smoke) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-backend/tests/laplacian_test.rs` — stubs for PRIM-09 (value + zero-degree + memory gate)
- [ ] `crates/mlrs-algos/tests/spectral_embedding_test.rs` — stubs for SPECTRAL-01 (rbf + knn + subspace + reject-oversize)
- [ ] `crates/mlrs-algos/tests/spectral_clustering_test.rs` — stubs for SPECTRAL-02 (label_perm)
- [ ] `crates/mlrs-py/tests/` spectral smoke — stubs for PY-06 share
- [ ] `prims/laplacian.rs` compiling stub (signature + geometry validation real, compute `todo!()`) + `mlrs-kernels` map-kernel stub — mirror the 08-01 `kernel_matrix.rs` Wave-0 scaffold
- [ ] `cluster/spectral_*.rs` + `estimators/spectral.rs` module homes registered (empty compiling stubs)
- [ ] `AlgoError::NSamplesExceedsMaxDim` variant + (optional) `n_neighbors` guard added to `error.rs`
- [ ] `scripts/gen_oracle.py` extended with `gen_spectral_embedding` / `gen_spectral_clustering` (committed `.npz` blobs)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm f32 documented-band confirmation for `embedding_` | SPECTRAL-01 | rocm GPU is opportunistic (not in default cpu CI gate) | Run `cargo test --features rocm -p mlrs-algos spectral_embedding_test` on gfx1100; confirm embedding within documented f32 band (or subspace test for degenerate spectra) |

*All other phase behaviors have automated cpu verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
