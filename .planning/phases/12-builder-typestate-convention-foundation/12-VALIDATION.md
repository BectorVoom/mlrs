---
phase: 12
slug: builder-typestate-convention-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-23
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (integration tests in `crates/*/tests/`, AGENTS.md §2) + `trybuild` for compile-fail |
| **Config file** | none — cargo convention; `tests/` dir per AGENTS.md §2 |
| **Quick run command** | `cargo test -p mlrs-algos --features cpu umap_test` (per-shell, targeted) + the trybuild ui test |
| **Full suite command** | `cargo test -p mlrs-algos -p mlrs-py --features cpu` (full algos suite ~6min — background it, gate targeted) |
| **Estimated runtime** | ~5s targeted per shell; ~6min full algos suite (reduce_test/svd_test dominate, unrelated to this phase) |

---

## Sampling Rate

- **After every task commit:** Run the targeted per-shell test (`cargo test -p mlrs-algos --features cpu <shell>_test`) + the trybuild ui test.
- **After every plan wave:** Run `cargo test -p mlrs-algos --features cpu` + `cargo test -p mlrs-py --features cpu` — confirms Success Criterion 3 (all 35 existing `any_estimator!` call sites green) on every merge.
- **Before `/gsd-verify-work`:** Full suite green (background the ~6min algos run), plus f32 round-trip under `--features rocm` for both shells (f64-on-rocm SKIPS-with-log).
- **Max feedback latency:** ~5 seconds (targeted) / ~6 min (full algos, backgrounded).

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TS-typestate | typestate | 1 | BLDR-02 | — | N/A | unit | `cargo test -p mlrs-algos --features cpu typestate` | ❌ W0 | ⬜ pending |
| TS-defaults-eq | umap/hdbscan | 2 | BLDR-01 | V5 | hyperparam validated at build() | unit | `cargo test -p mlrs-algos --features cpu umap::defaults_equal` | ❌ W0 (`tests/umap_test.rs`) | ⬜ pending |
| TS-build-reject | umap/hdbscan | 2 | BLDR-01 | T-12 / V5 | bad `min_dist`/`min_cluster_size` → typed `BuildError` | unit | `cargo test -p mlrs-algos --features cpu umap::build_rejects_bad_min_dist` | ❌ W0 | ⬜ pending |
| TS-compile-fail | compile-fail | 2 | BLDR-02 | — | predict/transform on `T<Unfit>` won't compile | compile-fail | `cargo test -p mlrs-algos --features cpu ui_predict_before_fit` | ❌ W0 (`tests/compile_fail.rs` + `tests/ui/*`) | ⬜ pending |
| TS-fit-roundtrip | umap/hdbscan | 2 | BLDR-02 | T-12 | fixed-size alloc from validated shape | unit | `cargo test -p mlrs-algos --features cpu umap::fit_roundtrip` | ❌ W0 | ⬜ pending |
| TS-no-leak | umap/hdbscan | 2 | BLDR-02 | T-12 / DoS | `live_bytes` no-climb across re-fit (memory gate) | unit | `cargo test -p mlrs-algos --features cpu umap::fit_no_leak` | ❌ W0 | ⬜ pending |
| TS-existing-green | pyo3 | 3 | BLDR-04 | — | every existing `any_estimator!` suite green (SC-3) | regression | `cargo test -p mlrs-py --features cpu` | ✅ (35 call sites) | ⬜ pending |
| TS-pyo3-smoke | pyo3 | 3 | BLDR-04 | — | shell instantiates in `Unfit` arm (no interpreter) | unit | `cargo test -p mlrs-py --features cpu manifold::unfit_default` | ❌ W0 | ⬜ pending |
| TS-not-fitted | pyo3 | 3 | BLDR-04 | V5 | accessor on `Unfit` arm → `not_fitted()` `PyValueError` | unit | `cargo test -p mlrs-py --features cpu manifold::not_fitted_before_fit` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-algos/tests/umap_test.rs` — BLDR-01/02 (defaults equality, build rejects, fit round-trip, no-leak)
- [ ] `crates/mlrs-algos/tests/hdbscan_test.rs` — same for HDBSCAN
- [ ] `crates/mlrs-algos/tests/compile_fail.rs` + `tests/ui/{predict,transform}_before_fit.rs` + `.stderr` — BLDR-02 structural proof
- [ ] `crates/mlrs-py/tests/manifold_test.rs` (or extend an existing py test) — `unfit_default` smoke + `not_fitted` runtime analog
- [ ] `trybuild = "1.0.117"` added to `crates/mlrs-algos` `[dev-dependencies]`
- [ ] (no new framework install — cargo + trybuild only)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live `estimator_checks` / `check_estimator` on the PyO3 shells | BLDR-04 (informational) | Needs a maturin + pyarrow host this environment lacks (per project MEMORY) | Deferred to UAT; Rust-side `unfit_default` + `not_fitted` smoke tests compensate |
| f32 round-trip under `--features rocm` | BLDR-02 | rocm is the runnable GPU gate; opportunistic | `cargo test -p mlrs-algos --features rocm umap::fit_roundtrip` (f64 SKIPS-with-log) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 6min (full algos, backgrounded) / 5s (targeted)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
