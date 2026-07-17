# Sources — mlrs Metrics Surface

Evidence ledger for `SPEC.md` / `PLAN.md`. Labels: VERIFIED(CODEGRAPH/LOCAL/WEB) · INFERRED · UNVERIFIED.

## Research reports (authoritative, in this folder's parent)
- `../RESEARCH.md` — ground-truth estimator inventory + coverage-gap analysis (companion; PY-ENSEMBLE recommendation NOT chosen).
- `../RESEARCH-METRICS.md` — metrics-surface deep dive: existing-code reuse, host-only module layout, PyO3 free-fn binding, oracle/fixture convention, files-to-touch, validation, risks. **Primary source for this plan.**

## Repository evidence
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs:160` — existing `accuracy_score(&[i32],&[i32])->f64` (the only pre-existing metric; reuse seam). [VERIFIED: CODEGRAPH]
- `crates/mlrs-algos/src/covariance/empirical_covariance.rs:414-427` — host f64-accumulate-then-cast precedent. [VERIFIED: LOCAL]
- `crates/mlrs-py/src/estimators/projection.rs:379-382` — `johnson_lindenstrauss_min_dim` `#[pyfunction]` precedent. [VERIFIED: LOCAL]
- `crates/mlrs-py/src/lib.rs:166-169,196,238` — `backend_supports_f64` `#[pyfunction]` + `m.add_function(wrap_pyfunction!(...))` registration. [VERIFIED: LOCAL]
- `crates/mlrs-py/src/ingress.rs:112-118` — arrow ingress is float-only (why plain-`Vec` extraction is used for integer labels). [VERIFIED: LOCAL]
- `crates/mlrs-py/src/egress.rs:32-68` — device-oriented egress (NOT used for host metrics). [VERIFIED: LOCAL]
- `crates/mlrs-py/python/mlrs/base.py:28-117`, `__init__.py:22-143` — MlrsBase + lazy `_ext`/`_load_ext`; namespace. [VERIFIED: LOCAL]
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-70` — oracle-replay + `_atol` template. [VERIFIED: LOCAL]
- `crates/mlrs-py/python/tests/{test_params,test_shims,test_estimator_checks}.py` — estimator-enumerating gates; **exempt** for free functions. [VERIFIED: LOCAL]
- `scripts/gen_oracle.py:15-17,41` — regen venv instructions + `_FIXTURE_DIR`; `np.savez` fixture pattern; no metrics generator yet. [VERIFIED: LOCAL]
- `crates/mlrs-algos/tests/random_forest_classifier_test.rs:87-91,202` — `load_npz`/`expect_f64`/`skip_f64_with_log` test template. [VERIFIED: LOCAL]
- `.planning/ROADMAP.md:216-231` — Phase 24 metrics success criterion (METR-01/02/03) + mandatory degenerate fixtures. [VERIFIED: LOCAL]

## Versions
- `pyo3 0.28.3` (pinned — do NOT bump), `arrow 59.0.0`, Rust `stable`, Python ≥3.12 (`abi3-py312`). [VERIFIED: LOCAL Cargo.lock, rust-toolchain.toml]
- Oracle: `numpy scipy scikit-learn` venv. Exact sklearn version producing fixtures — **UNVERIFIED**, must be stamped (Q6).

## External
- scikit-learn `sklearn.metrics` stable API docs (accessed 2026-07-16): signatures, `average`/`zero_division` defaults, `mean_squared_error` `squared=` deprecation, `log_loss` eps clipping, `roc_auc_score` `multi_class` + single-class `ValueError`. [VERIFIED: WEB]

## Validation commands (verified against repo)
- Rust algos: `cargo test -p mlrs-algos --features cpu` (f64 gate), `--features wgpu` (f32 gate).
- PyO3 integration: `cargo test -p mlrs-py --features cpu`.
- Python shim/oracle: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml` then `pytest crates/mlrs-py/python/tests/`.
- Fixture regen: `python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn && /tmp/oracle-venv/bin/python scripts/gen_oracle.py`.
- No justfile/Makefile/CI in repo.
