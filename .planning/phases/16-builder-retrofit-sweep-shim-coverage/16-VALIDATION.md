---
phase: 16
slug: builder-retrofit-sweep-shim-coverage
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-24
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 16-RESEARCH.md § Validation Architecture (HIGH confidence, file-verified).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework (Rust)** | `cargo test` integration tests in `crates/*/tests/` (AGENTS.md §2 — NO in-source `#[cfg(test)]`; verified zero in src) |
| **Framework (Python)** | `pytest` in `crates/mlrs-py/python/tests/` (runs without compiled `_mlrs`) |
| **Config file** | workspace `Cargo.toml`; per-test oracle fixtures (committed `.npz` blobs) |
| **Quick run command** | `cargo test --features cpu --test <estimator>_test` (per-estimator gate) |
| **Full suite command** | `cargo test --features cpu` (targeted set — see warning) |
| **Estimated runtime** | per-estimator suite ~seconds–minutes; full run ~6min+ (slow, can exhaust disk — prefer targeted) |

> **Memory-backed warnings:** full `cargo test --features cpu` is slow (~6min+) and can exhaust the shared `/` partition (ENOSPC bricks Bash). Run **targeted** per-estimator / per-module gates; reserve a full run for a controlled phase-end pass. Worktree isolation is broken in this env → sequential, non-worktree execution (this phase is parallel-unsafe by design). Python live FFI (`maturin`+`pyarrow`) is unavailable → static shim gate is the maximum verifiable; live `check_estimator` routes to UAT.

---

## Sampling Rate

- **After every task commit:** the migrated estimator's `cargo test --features cpu --test <est>_test` **plus** `cargo build -p mlrs-py --features cpu` (catches the hidden PyO3 cross-crate break — the load-bearing risk per RESEARCH §Pitfall 3).
- **After every plan wave (module):** that module's suites + `cargo build` on **both** crates (`mlrs-algos`, `mlrs-py`).
- **Before `/gsd-verify-work`:** targeted `cargo test --features cpu` green + `compile_fail` (trybuild) green + full `pytest` shim suite green + the `traits.rs`-gone grep returns empty.
- **GPU gate (per project memory):** `--features rocm` runs the f32 GPU path (gfx1100 / ROCm 7.1.1); f64-on-rocm SKIPs-with-log. cuda compile-only/untestable here.
- **Max feedback latency:** one per-estimator suite run (~seconds–low minutes).

---

## Per-Task Verification Map

> Filled per-PLAN by the planner. Each estimator migration task maps to its oracle suite; each shim/PyO3 task maps to the Python static gate or a Rust unit test. Skeleton rows below show the shape; planner expands to one row per task.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 16-00-01 | 00 | 0 | BLDR-03 | — | `typestate.rs` grows the 5 missing accessor traits + `Transform::inverse_transform` default | unit/build | `cargo build -p mlrs-algos --features cpu` | ✅ | ⬜ pending |
| 16-00-02 | 00 | 0 | SHIM-01 | — | AST `__init__`-purity assertion exists (NEW — no `import ast` today) | python | `pytest crates/mlrs-py/python/tests/test_params.py -k init_purity` | ❌ W0 | ⬜ pending |
| 16-01-01 | 01 | 1 | BLDR-03 | T-16-V5 | `validate_geometry` stays atop ported `fit`; data-indep validation → `BuildError` | unit/oracle | `cargo test --features cpu --test ridge_test` | ✅ | ⬜ pending |
| 16-NN-NN | NN | N | BLDR-03 | T-16-V5 | predict-before-fit is a compile error | compile-fail | `cargo test --features cpu --test compile_fail` | ✅ harness; ⚠️ add fixtures | ⬜ pending |
| 16-ZZ-ZZ | ZZ | final | BLDR-03 | — | `crate::traits` / `mlrs_algos::traits` fully removed | grep gate | `! grep -rq 'mlrs_algos::traits\|crate::traits' crates/` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-algos/src/typestate.rs` — ADD `PredictLabels`, `KNeighbors`, `ScoreSamples`, `PredictProba`, `PredictLogProba` + `Transform::inverse_transform` default. **Blocks every estimator implementing those traits.** Gate: `cargo build -p mlrs-algos --features cpu` + `typestate_test.rs`.
- [ ] `crates/mlrs-py/python/tests/test_params.py` — ADD AST-based `__init__`-purity test (`import ast`, `inspect.getsource`) — D-07 step 3, does **not** exist today.
- [ ] `test_shims.py` / `test_params.py` / `test_estimator_checks.py` — replace hard-coded `ALL_12` with the full in-scope shim set; add `EXPECTED_PARAMS`/`SET_PARAM` rows for every new class incl. UMAP/HDBSCAN.
- [ ] New pure-Python shim modules/classes for the missing estimators (mechanical, `MlrsBase` template) — exact in-scope list reconciled by the planner against `EXPECTED_PARAMS` (RESEARCH A6).
- [ ] `tests/ui/` — per-estimator-family compile-fail fixtures (or accept the UMAP fixture as the representative typestate proof).
- [ ] PyUMAP: add `transform`/`fit_transform` `#[pymethods]`; PyHDBSCAN: add `fit_predict`/`probabilities_`/`outlier_scores_` `#[pymethods]` (VERIFIED missing).
- [ ] Lock the builder-setter type convention (**`f64` setters**, uniform with shipped mbsgd/umap builders; `build::<F>()` narrows) before the sweep.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live sklearn `check_estimator` / `estimator_checks` over the compiled extension | SHIM-03 | No `maturin`+`pyarrow` host in this environment (Python wheel untestable in env) | UAT: on a host with the built `_mlrs` wheel + sklearn, run the live `parametrize_with_checks` suite against each estimator and confirm no unexpected failures beyond the by-design xfail map |

*All other phase behaviors have automated verification (static Python gate + Rust oracle suites).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (5 typestate traits, AST purity test, PyO3 method gaps)
- [ ] No watch-mode flags
- [ ] Feedback latency < one per-estimator suite run
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
