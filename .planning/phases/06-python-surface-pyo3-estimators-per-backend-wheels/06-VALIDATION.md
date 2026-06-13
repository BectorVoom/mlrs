---
phase: 6
slug: python-surface-pyo3-estimators-per-backend-wheels
status: draft
nyquist_compliant: false
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
| {N}-01-01 | 01 | 0 | PY-{XX} | T-6-NN / — | {expected secure behavior or "N/A"} | unit | `{command}` | ❌ W0 | ⬜ pending |

*The planner populates one row per task from the produced PLAN.md files. Map source: `06-RESEARCH.md` → Validation Architecture → "Phase Requirements → Test Map".*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

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
