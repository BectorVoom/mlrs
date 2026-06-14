---
phase: 6
slug: python-surface-pyo3-estimators-per-backend-wheels
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-13
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source of truth for the Wave-0 test scaffold and the per-task verify map is
> the **Validation Architecture** section of `06-RESEARCH.md`. The planner fills
> the per-task map below from the plan it produces.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust, existing `crates/mlrs-py/tests/`) + `pytest` (Python, NEW — Wave 0) |
| **Config file** | none yet for pytest — add `python/tests/conftest.py` (Wave 0) |
| **Quick run command** | `cargo test -p mlrs-py --features cpu` (Rust); `pytest python/tests -x -k <name>` (Python, after `maturin develop --features cpu`) |
| **Full suite command** | `maturin develop --features cpu && pytest python/tests`, then repeat `--features rocm` (f32 subset only) |
| **Estimated runtime** | ~TBD seconds (planner to estimate; mlrs-backend cpu suite is slow — run targeted gates per task) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p mlrs-py --features cpu` + targeted `pytest -k <name>`
- **After every plan wave:** Run `maturin develop --features cpu && pytest python/tests` (cpu f64); `--features rocm` for the f32 subset
- **Before `/gsd-verify-work`:** Full suite must be green — Python oracle + relevant `estimator_checks` on cpu(f64); f32 subset on rocm; all four wheels build with correct names; absent-driver import test passes
- **Max feedback latency:** TBD seconds (planner to set)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | deps (PY-04) | — | Dep floors are legitimate published versions (supply-chain) | manual gate | checkpoint — package-legitimacy review (no auto verify) | ❌ W1 | ⬜ pending |
| 06-01-02 | 01 | 1 | PY-03, PY-05 | T-06-10 | Exactly one PyO3 (0.28) links the cdylib; no second ABI | build | `cargo tree -p mlrs-py --features cpu -i pyo3 \| grep 'pyo3 v0.28'` (and no v0.29) | ❌ W1 | ⬜ pending |
| 06-01-03 | 01 | 1 | PY-04 | — | Four pyproject templates, constant `module-name="mlrs._mlrs"` | source | `test -f .../{cpu,wgpu,cuda,rocm}.pyproject.toml` + 4× module-name | ❌ W1 | ⬜ pending |
| 06-01-04 | 01 | 1 | PY-01 | — | Pure-Python skeleton AST-parses (12 estimator shells) | source | `python3 -c "ast.parse(...mlrs/*.py)"` → `parse-ok` | ❌ W1 | ⬜ pending |
| 06-01-05 | 01 | 1 | PY-03 | T-06-11 | pytest scaffold parses; arrow `FromPyArrow` symbol confirmed | source | conftest ast-ok + `grep from_pyarrow(_bound)? arrow_symbol_probe.rs` | ❌ W1 | ⬜ pending |
| 06-02-01 | 02 | 2 | PY-03, PY-05 | T-06-11 | Owned PyCapsule ingress (no `&[u8]` borrow); contiguity hard-reject; f64-on-incapable guard | integration (Rust) | `cargo test -p mlrs-py --features cpu --test ingress_test` | ❌ W1 | ⬜ pending |
| 06-02-02 | 02 | 2 | PY-04 | T-06-05 / T-06-16 | `catch_unwind` → `PyImportError` on absent driver; panic does not cross FFI | integration (Rust) | `cargo test -p mlrs-py --features cpu --test probe_test` | ❌ W1 | ⬜ pending |
| 06-03-01 | 03 | 3 | PY-01, PY-02 | — | Linear+decomposition `#[pyclass]` wrappers; sklearn-named ctors | build | `cargo build -p mlrs-py --features cpu` | ❌ W1 | ⬜ pending |
| 06-03-02 | 03 | 3 | PY-01, PY-02 | — | Cluster+neighbors wrappers; i32 labels/indices; predict-less DBSCAN/NN | build | `cargo build -p mlrs-py --features cpu` | ❌ W1 | ⬜ pending |
| 06-03-03 | 03 | 3 | PY-01 | T-06-10 | All 12 pyclasses registered + construct (GIL released in compute) | integration (Rust) | `cargo test -p mlrs-py --features cpu --test pyclass_smoke_test` | ❌ W1 | ⬜ pending |
| 06-04-01 | 04 | 4 | PY-03 | T-06-11, T-06-12 | `normalize_X` → fresh contiguous pyarrow + (rows,cols); finite-check | integration | `import mlrs; normalize_X(np.eye(3,f32))` → `io-ok` | ❌ W1 | ⬜ pending |
| 06-04-02 | 04 | 4 | PY-01 | — | 12 sklearn-compatible shims; `fit` returns self; clone works | integration | `import mlrs; clone(...); hasattr(LogisticRegression,'C')` → `shim-ok` | ❌ W1 | ⬜ pending |
| 06-04-03 | 04 | 4 | PY-02 | — | get_params/set_params round-trip; sklearn names + defaults | unit (pytest) | `pytest .../test_params.py` | ❌ W1 | ⬜ pending |
| 06-05-01 | 05 | 5 | PY-01 | — | 1e-5 oracle through full numpy→PyCapsule→device→numpy path (4 families) | integration (pytest) | `pytest .../test_oracle_{linear,cluster,decomposition,neighbors}.py` | ❌ W1 | ⬜ pending |
| 06-05-02 | 05 | 5 | PY-05, PY-03 | T-06-11 | f32/f64 dispatch; f64-on-rocm raises (no downcast); GIL-release smoke | unit (pytest) | `pytest .../test_dtype.py` | ❌ W1 | ⬜ pending |
| 06-06-01 | 06 | 5 | PY-01 | — | Relevant `estimator_checks` subset passes; skips documented | integration (pytest) | `pytest .../test_estimator_checks.py` + `checks_triage.md` exists | ❌ W1 | ⬜ pending |
| 06-06-02 | 06 | 5 | PY-04 | T-06-05 / T-06-16 | Four wheels, distinct dist names, abi3-py312; driver-absent `ImportError` | build/smoke | `maturin build -m .../cpu.pyproject.toml` + wheel-name assert | ❌ W1 | ⬜ pending |
| 06-06-03 | 06 | 5 | PY-04 | T-06-16 | cuda wheel build + driver-absent import (untestable backend) | manual | human-verify (cuda compile-only in this env) | ❌ W1 | ⬜ pending |

*Map source: the six `06-0N-PLAN.md` files' `<verify><automated>` blocks. Scaffold (`tests/*.py`, pyproject templates) is created during Wave-1 execution of Plan 06-01, hence `File Exists ❌ W1` until then; `wave_0_complete` flips true after 06-01 lands.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky · Threat refs map to each plan's `<threat_model>` (T-06-05/16 driver-absent abort, T-06-10 FFI panic, T-06-11 sliced-array aliasing, T-06-12 NaN/inf).*

---

## Wave 0 Requirements

- [ ] `crates/mlrs-py/Cargo.toml` — pyo3 0.28 (abi3-py312, extension-module) + arrow pyarrow + mlrs-algos/mlrs-backend + backend features
- [ ] `[workspace.dependencies]` — add `pyo3 = "0.28"` (with the ABI-pin comment; locked-by-compat, NOT latest)
- [ ] `crates/mlrs-py/python/mlrs/` — pure-Python shim package skeleton (base + per-family modules)
- [ ] `pyproject/{cpu,wgpu,cuda,rocm}.pyproject.toml` — per-backend templated configs (`[project].name`, `features`, `module-name="mlrs._mlrs"`, `python-source`)
- [ ] `python/tests/conftest.py` — fixture loader + sign-flip/label-perm helpers + capability skip marker
- [ ] `python/tests/test_oracle_*.py`, `test_estimator_checks.py`, `test_dtype.py`, `test_params.py` — stubs for PY-01..PY-05
- [ ] Framework install: `pip install maturin pyarrow scikit-learn numpy pytest` into a venv (PEP 668)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| cuda wheel build / driver-absent import | PY-04 | cuda is compile-only / untestable in this environment | Build `mlrs-cuda` wheel; confirm name + abi3-py312; driver-absent `ImportError` verified opportunistically |
| f64-on-rocm device behavior | PY-05 | rocm f64 is unsupported (cubecl-cpp 0.10 / HIP) — D-04 raises before device | Covered by automated raise-test; device-level confirmation manual/opportunistic |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < TBD s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
