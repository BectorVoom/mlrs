---
phase: 04
slug: closed-form-estimators
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-12
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` / `cargo test` (no external runner); tests live in `crates/*/tests/` per AGENTS.md §2 (no in-source `mod tests`) |
| **Config file** | none — `cargo test` per crate with backend feature flags; Wave 0 (04-01) installs the test scaffolds + committed `.npz` fixtures |
| **Quick run command** | `cargo test -p mlrs-algos --features cpu` (f64 gate; f64 runs on cpu) |
| **Full suite command** | `cargo test -p mlrs-algos --features cpu && cargo test -p mlrs-algos --features rocm && cargo test -p mlrs-backend --features cpu --test cholesky_test && cargo test -p mlrs-backend --features rocm --test cholesky_test && cargo test -p mlrs-backend --features cpu --test memory_gate_test` |
| **Estimated runtime** | quick ~30–60 s; full suite ~3–5 min (two backends × algos + cholesky + memory gate; f64-on-rocm skips-with-log) |

**Gate note (D-07, supersedes ROADMAP cpu+wgpu):** f64 validates on **cpu** (`f64_supported=true`); f32 validates on **rocm** (gfx1100, ROCm 7.1.1); f64-on-rocm **skips-with-log** via `skip_f64_with_log` (cubecl-cpp 0.10 does not register F64 for HIP — EXPECTED). wgpu is opportunistic only.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate> --features cpu --test <relevant_test>`
- **After every plan wave:** Run `cargo test -p mlrs-algos --features cpu` (+ `--features rocm` for the f32 gate)
- **Before `/gsd-verify-work`:** Full suite must be green on cpu(f64)+rocm(f32); f64 tests skip-with-log on rocm
- **Max feedback latency:** ~60 seconds (quick run)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | LINEAR-01/02, DECOMP-01/02 | T-04-01-01 / T-04-01-02 | `AlgoError` rejects `n_components > min(m,n)`; `PrimError::NotPositiveDefinite` variant defined for negative-pivot (no silent NaN) | build/unit | `cargo build -p mlrs-algos --features cpu && cargo build -p mlrs-algos --features rocm` | ✅ (creates) | ⬜ pending |
| 04-01-02 | 01 | 1 | LINEAR-01/02, DECOMP-01/02 | — | Committed `.npz` fixtures are trusted repo build artifacts; `mlrs_core::load_npz` validates keys/shape on load | unit (fixture) | `cargo test -p mlrs-backend --features cpu --test cholesky_test fixture_loads -- --ignored` (loads one fixture, asserts keys/shape) | ✅ (creates) | ⬜ pending |
| 04-01-03 | 01 | 1 | LINEAR-01/02, DECOMP-01/02 | — | N/A (compiling `#[ignore]` Nyquist stubs; no device launch yet) | unit (stub) | `cargo test -p mlrs-backend --features cpu --test cholesky_test -- --list && cargo test -p mlrs-algos --features cpu --test pca_test -- --list` | ✅ (creates) | ⬜ pending |
| 04-02-01 | 02 | 2 | LINEAR-02 | T-04-02-02 | Cholesky diagonal sqrt-arg guard sets `info_out` flag → no NaN | unit (build) | `cargo build -p mlrs-kernels --features cpu && cargo build -p mlrs-kernels --features rocm` | ❌ W0 (04-01) | ⬜ pending |
| 04-02-02 | 02 | 2 | LINEAR-02 | T-04-02-01 / T-04-02-02 / T-04-02-03 | Geometry validated before `unsafe` launch (NotSquare/ShapeMismatch); non-SPD → `PrimError::NotPositiveDefinite`; `‖L·Lᵀ−A‖` & `‖A·x−b‖` invariants | unit (invariant + scipy oracle) | `cargo test -p mlrs-backend --features cpu --test cholesky_test && cargo test -p mlrs-backend --features rocm --test cholesky_test` | ❌ W0 (04-01) | ⬜ pending |
| 04-03-* | 03 | 3 | LINEAR-01 | T-04-03-* | Small-σ cutoff (`σ⁺=0` below threshold) prevents inf/NaN on rank-deficient X; geometry validated | oracle | `cargo test -p mlrs-algos --features cpu --test linear_regression_test && cargo test -p mlrs-algos --features rocm --test linear_regression_test` | ❌ W0 (04-01) | ⬜ pending |
| 04-04-* | 04 | 3 | DECOMP-01, DECOMP-02 | T-04-04-* | `n_components ≤ min(m,n)` validated at fit; align_rows canonicalization before compare | oracle | `cargo test -p mlrs-algos --features cpu --test pca_test && cargo test -p mlrs-algos --features cpu --test truncated_svd_test && cargo test -p mlrs-algos --features rocm --test pca_test && cargo test -p mlrs-algos --features rocm --test truncated_svd_test` | ❌ W0 (04-01) | ⬜ pending |
| 04-05-* | 05 | 4 | LINEAR-02 | T-04-05-* | Ridge consumes validated Cholesky prim; intercept not penalized; fit→predict memory gate (Gram/factor buffer reuse, D-11 gate 2) | oracle + gate | `cargo test -p mlrs-algos --features cpu --test ridge_test && cargo test -p mlrs-algos --features rocm --test ridge_test && cargo test -p mlrs-backend --features cpu --test memory_gate_test` | ❌ W0 (04-01) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Note: plans 04-03/04/05 each contain multiple oracle test functions (f32+f64 per attribute / per alpha or shape); the row collapses them to one command per plan. f64 functions carry the `skip_f64_with_log` gate (run on cpu, skip-with-log on rocm).*

---

## Wave 0 Requirements

Created by **04-01** (the Wave-0 scaffold plan) as compiling `#[ignore]` Nyquist stubs + committed fixtures, so every downstream test exists before its implementation lands:

- [ ] `crates/mlrs-backend/tests/cholesky_test.rs` — stubs for the new Cholesky primitive (`‖L·Lᵀ−A‖`, `‖A·x−b‖`, NotPositiveDefinite guard); activated by 04-02
- [ ] `crates/mlrs-algos/tests/linear_regression_test.rs` — stubs for LINEAR-01 (coef/intercept/predict + collinear cutoff); activated by 04-03
- [ ] `crates/mlrs-algos/tests/ridge_test.rs` — stubs for LINEAR-02 (coef/intercept across alpha sweep, intercept not penalized); activated by 04-05
- [ ] `crates/mlrs-algos/tests/pca_test.rs` — stubs for DECOMP-01 (all attrs + transform/inverse_transform after align_rows); activated by 04-04
- [ ] `crates/mlrs-algos/tests/truncated_svd_test.rs` — stubs for DECOMP-02 (attrs + transform vs arpack after align_rows); activated by 04-04
- [ ] Memory-gate extension — extend `crates/mlrs-backend/tests/memory_gate_test.rs` for the fit→predict/transform pipeline (D-03 / D-11 gate 2 Gram/factor buffer reuse); activated by 04-05
- [ ] Shared fixtures (conftest analog) — committed `.npz` blobs under `tests/fixtures/` (`cholesky_{f32,f64}_seed42.npz`, `linear_regression_{f32,f64}_seed42.npz`, `ridge_{f32,f64}_seed42.npz`, `pca_{f32,f64}_seed42.npz`, `truncated_svd_{f32,f64}_seed42.npz`) generated by the new `scripts/gen_oracle.py` generators (`gen_cholesky`, `gen_linear_regression`, `gen_ridge`, `gen_pca`, `gen_truncated_svd`); fixtures are committed artifacts, never regenerated in CI. The `fixture()` resolver helper (copied from `svd_test.rs`) is the per-crate shared loader — Rust has no `conftest.py`; the resolver + `mlrs_core::load_npz` play that role.

*Rust has no framework install step — `cargo test` is built in. The "install" Wave 0 owns is adding `mlrs-backend`/`mlrs-core`/`cubecl` deps to `mlrs-algos/Cargo.toml` (currently only `thiserror`) so the test crates compile.*

---

## Manual-Only Verifications

All phase behaviors have automated verification. Numerical correctness is checked against committed scikit-learn/scipy `.npz` fixtures via `assert_close` (1e-5 abs+rel); algebraic invariants (`‖L·Lᵀ−A‖`, `‖A·x−b‖`) are host-checked after read-back; geometry/typed-error rejection is asserted directly; the memory contract is a build-failing PoolStats gate. No visual, interactive, or human-judgment behavior exists in this Rust-only numerical phase (the Python surface is Phase 6).

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (5 test stubs + memory-gate extension + committed fixtures)
- [x] No watch-mode flags
- [x] Feedback latency < 60s (quick run)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-12
