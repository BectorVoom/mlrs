---
phase: 7
slug: covariance-projection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-14
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `07-RESEARCH.md` §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (cargo test); tests in `crates/*/tests/` (AGENTS.md §2 — never in-source `mod tests`) |
| **Config file** | none — cargo test; feature-gated by `--features cpu` / `--features rocm` |
| **Quick run command** | `cargo test --features cpu -p <crate> <test_name>` (targeted) |
| **Full suite command** | `cargo test --features cpu` then `cargo test --features rocm` (the two correctness gates) |
| **Estimated runtime** | targeted ~secs; full cpu suite ~6 min (reduce_test 248s, svd_test 99s — keep new SVD fixtures tiny) |

---

## Sampling Rate

- **After every task commit:** Run the targeted test for the file touched — `cargo test --features cpu -p <crate> <name>`
- **After every plan wave:** Run the phase's new tests on cpu (`-p mlrs-algos` + the two new `-p mlrs-backend` prim tests); background the full cpu suite (~6 min)
- **Before `/gsd-verify-work`:** Full cpu suite green, then `cargo test --features rocm` for the f32 bands (f64 skips-with-log on rocm)
- **Max feedback latency:** ~60s (targeted); full suite backgrounded

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| PRIM-06 | Gaussian matrix stats (mean≈0, var≈1/n_components) + seed-reproducibility (same seed → identical matrix across runs/backends) + Achlioptas density/value stats + Fisher-Yates permutation bijection | unit (distribution + repro) | `cargo test --features cpu -p mlrs-backend rng_` | ❌ W0 |
| PRIM-06 | PoolStats memory gate for `rng.rs` (host-generate + single upload; bounded allocations) | unit (pool) | `cargo test --features cpu -p mlrs-backend rng_memory_gate` | ❌ W0 |
| PRIM-07 | 2+-batch merge vs host reference; ddof=1; svd_flip applied; f64 1e-5 / f32 band | unit (oracle + multi-batch) | `cargo test --features cpu -p mlrs-backend incremental_svd_` | ❌ W0 |
| PRIM-07 | PoolStats memory gate for `incremental_svd.rs` | unit (pool) | `cargo test --features cpu -p mlrs-backend incremental_svd_memory_gate` | ❌ W0 |
| COV-01 | `covariance_`/`location_`/`precision_` vs sklearn EmpiricalCovariance, 2 sizes incl. rank-deficient (n≤p) for precision_ | oracle (1e-5) | `cargo test --features cpu -p mlrs-algos empirical_covariance_` | ❌ W0 |
| COV-02 | `shrinkage_` (∈[0,1]) + `covariance_` vs sklearn LedoitWolf, across two `n` | oracle (1e-5) | `cargo test --features cpu -p mlrs-algos ledoit_wolf_` | ❌ W0 |
| DECOMP-03 | all attrs + `transform`/`inverse_transform` vs sklearn IncrementalPCA, via `partial_fit` over batches AND via `fit()`; whiten on/off | oracle (1e-5, post align_rows) | `cargo test --features cpu -p mlrs-algos incremental_pca_` | ❌ W0 |
| PROJ-01/02 | property gate: JL distortion (averaged, strict), matrix moment stats, seed-repro across backends, `transform==X·componentsᵀ`; `johnson_lindenstrauss_min_dim` value-matched | property + 1 value-oracle | `cargo test --features cpu -p mlrs-algos random_projection_` | ❌ W0 |
| (recurring) | every f64 oracle case gated by `skip_f64_with_log` | gate | (in each test) | pattern exists (gemm_test.rs) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-backend/tests/rng_test.rs` — PRIM-06 distribution + seed-repro + Achlioptas + permutation + memory gate
- [ ] `crates/mlrs-backend/tests/incremental_svd_test.rs` — PRIM-07 2+-batch merge + memory gate
- [ ] `crates/mlrs-algos/tests/empirical_covariance_test.rs` — COV-01 (incl. rank-deficient precision_)
- [ ] `crates/mlrs-algos/tests/ledoit_wolf_test.rs` — COV-02 (two n)
- [ ] `crates/mlrs-algos/tests/incremental_pca_test.rs` — DECOMP-03 (partial_fit + fit + whiten)
- [ ] `crates/mlrs-algos/tests/random_projection_test.rs` — PROJ-01/02 property gate + jl_min_dim value
- [ ] `scripts/gen_oracle.py` — 4 new generators (empirical_covariance, ledoit_wolf, incremental_pca, jl_min_dim) + `main()` wiring; regen in `/tmp` venv (PEP 668), commit blobs
- [ ] Framework install: none (cargo built-in)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| f32-on-rocm tolerance bands for LedoitWolf / IncrementalPCA | COV-02 / DECOMP-03 | Bands are Claude's-discretion, measured from the standalone prim run on actual rocm hardware | Run `cargo test --features rocm` on gfx1100/ROCm 7.1.1; record observed max abs/rel error; pin the documented band |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s (targeted)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
