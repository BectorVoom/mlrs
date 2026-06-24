---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 11
subsystem: mlrs-py (pure-Python sklearn shim)
tags: [shim, sklearn, SHIM-01, SHIM-03, ast-purity, estimator-checks, wave-11]
requires:
  - "MlrsBase machinery (base.py: _normalize/_ext/_suffixed/_post_fit/__sklearn_tags__) — complete"
  - "PyO3 wraps for all 15 estimators (linear/naive_bayes/kernel/spectral/manifold/cluster.rs) — shipped"
  - "Plan-10 PyUMAP transform/fit_transform + PyHDBSCAN fit_predict/probabilities_/outlier_scores_"
  - "Plan-00 test_init_purity_ast static SHIM-01 gate (parametrized over the shared shim list)"
provides:
  - "mlrs.LinearSVC / mlrs.LinearSVR / mlrs.MBSGDClassifier / mlrs.MBSGDRegressor shims"
  - "mlrs.GaussianNB / MultinomialNB / BernoulliNB / ComplementNB / CategoricalNB shims"
  - "mlrs.KernelRidge / mlrs.KernelDensity shims"
  - "mlrs.SpectralClustering / mlrs.SpectralEmbedding shims"
  - "mlrs.UMAP / mlrs.HDBSCAN shims (SHIM-01 pair)"
  - "Full 32-shim static test matrix (ALL_SHIMS derived from the exported set)"
affects:
  - "Plan 16-12 (traits.rs deletion / phase wrap-up — full Python estimator coverage now lands)"
tech-stack:
  added: []
  patterns:
    - "shared _BaseNB mixin holds the 5-NB common predict/predict_proba/predict_log_proba surface (concrete subclasses keep a PURE __init__ each)"
    - "random_state -> Rust seed mapping done in fit (SpectralClustering), NOT __init__ (keeps __init__ AST-pure)"
    - "ALL_SHIMS derived from {n in mlrs.__all__ if issubclass(getattr(mlrs,n), MlrsBase)} — the matrix cannot drift from the exported surface"
    - "test_matrix_covers_exports + test_fit_free_checks_never_xfailed as drift/regression guards"
key-files:
  created:
    - crates/mlrs-py/python/mlrs/naive_bayes.py
    - crates/mlrs-py/python/mlrs/kernel_ridge.py
    - crates/mlrs-py/python/mlrs/density.py
    - crates/mlrs-py/python/mlrs/manifold.py
  modified:
    - crates/mlrs-py/python/mlrs/linear.py
    - crates/mlrs-py/python/mlrs/cluster.py
    - crates/mlrs-py/python/mlrs/__init__.py
    - crates/mlrs-py/python/tests/test_shims.py
    - crates/mlrs-py/python/tests/test_params.py
    - crates/mlrs-py/python/tests/test_estimator_checks.py
decisions:
  - "Shim every PyO3-wrapped estimator (RESEARCH Open Q1) — the 15 missing classes added; full estimator parity reached (32 shims)"
  - "5-NB share a _BaseNB(ClassifierMixin, MlrsBase) for the predict surface; each concrete NB keeps its own PURE __init__ so the AST gate still applies per-class"
  - "SpectralClustering exposes sklearn random_state (mapped to the Rust seed inside fit, None->0) — mirrors KMeans; keeps __init__ pure"
  - "SpectralEmbedding/UMAP are TransformerMixin; SpectralEmbedding exposes fit_transform+embedding_ (no out-of-sample transform, matching sklearn); UMAP adds out-of-sample transform (Plan-10 wrap)"
  - "KernelDensity subclasses MlrsBase only (no family mixin) — its surface is fit + score_samples"
  - "Matrix expanded to the FULL exported set: the 15 new + 5 previously-untested pre-existing shims (IncrementalPCA, EmpiricalCovariance, LedoitWolf, Gaussian/SparseRandomProjection) -> 32 total"
metrics:
  duration: ~28m
  completed: 2026-06-24
  tasks: 3
  files: 10
  commits: 3
status: complete
---

# Phase 16 Plan 11: SHIM-01 + SHIM-03 — pure-Python sklearn shim completion Summary

Completed the pure-Python sklearn shim by adding a faithful `MlrsBase` subclass for every PyO3-wrapped estimator that lacked one (the 15 missing classes, including `mlrs.UMAP` and `mlrs.HDBSCAN`), then expanded the static test matrix to the **full 32-shim exported set** and exercised the Plan-00 AST-purity gate plus the fit-free `estimator_checks` subset over all of them. `get_params`/`set_params`/`clone` come free from `BaseEstimator` given the faithful zero-validation `__init__`s; the live `check_estimator` run (which needs the compiled `_mlrs` extension) stays deferred to UAT (SHIM-03 by-design).

## What Was Built

### Task 1 — 11 family shim classes (commit `19292af`)

| File | Classes | Mixin | Notes |
|------|---------|-------|-------|
| `linear.py` | LinearSVC, LinearSVR, MBSGDClassifier, MBSGDRegressor | Classifier/Regressor | sklearn `C` stored verbatim as `self.C`; classifiers derive `classes_` from the wrapper `classes_()` getter (no `n_classes()` on SVC/MBSGD); intercept is a scalar accessor |
| `naive_bayes.py` (new) | GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB | ClassifierMixin | shared `_BaseNB` holds predict/predict_proba/predict_log_proba; each concrete class keeps a PURE `__init__` |
| `kernel_ridge.py` (new) | KernelRidge | RegressorMixin | predict + `dual_coef_` |
| `density.py` (new) | KernelDensity | MlrsBase only | fit + `score_samples` (no predict; sklearn KernelDensity has no scoring mixin) |

Every `__init__` stores each ctor arg verbatim under the same sklearn-name with NO validation/computation, all args defaulted (zero-arg constructible — Pitfall 6). Defaults read directly from each `Py*` `#[new]` signature in `crates/mlrs-py/src/estimators/`.

### Task 2 — Spectral pair + `mlrs.UMAP` / `mlrs.HDBSCAN` (commit `175ae09`)

- `cluster.py`: **SpectralClustering** (`ClusterMixin`, labels-only; sklearn `random_state` mapped to the Rust `seed` inside `fit`, `None`->0), **SpectralEmbedding** (`TransformerMixin`; `fit_transform` + `embedding_`, no out-of-sample transform — matches sklearn), **HDBSCAN** (`ClusterMixin`; `labels_` + `probabilities_`/`outlier_scores_` forwarding to the Plan-10 getters, returning `None` until the GLOSH feature-space front-end lands).
- `manifold.py` (new): **UMAP** (`TransformerMixin`; 16 umap-learn-named params verbatim; `transform`/`fit_transform`/`embedding_` forwarding to the Plan-10 PyUMAP `transform_f{32,64}` / `fit_transform_f{32,64}` methods).
- `__init__.py`: imported + `__all__`-listed all 15 new shims.

### Task 3 — full static test-matrix expansion (commit `0fc3106`)

- `test_params.py` / `test_shims.py`: replaced the hard-coded `ALL_12` with **`ALL_SHIMS`**, derived from `{n in mlrs.__all__ if issubclass(getattr(mlrs,n), MlrsBase)}` so the matrix can never drift from the exported surface. Added `EXPECTED_PARAMS`/`SET_PARAM` rows for the 15 new classes **and the 5 previously-untested pre-existing shims** (IncrementalPCA, EmpiricalCovariance, LedoitWolf, Gaussian/SparseRandomProjection) — the matrix now spans the full **32** exported estimator shims.
- New `test_matrix_covers_exports` guards `EXPECTED_PARAMS`/`SET_PARAM` keys == exported shim set (no shim untested, no stale entry).
- The Plan-00 `test_init_purity_ast` now iterates all 32 shims automatically (it reads the shared `ALL_SHIMS`), confirmed green incl. UMAP/HDBSCAN.
- New family-surface assertions (`test_new_shim_family_surfaces`): transformers expose transform/fit_transform, cluster shims are labels-only (no standalone predict), classifiers/regressors expose predict, KernelDensity exposes score_samples.
- `test_estimator_checks.py`: 15 new estimators appended to the parametrized list with by-design xfail maps (reusing `_COMMON`/`_SUPERVISED`/`_CLASSIFIER`/`_N_ITER`). New `test_fit_free_checks_never_xfailed` proves the three fit-free checks (`check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance`) are NOT in any class's xfail map (A7 / D-07 step 4).

## Test Gates — what ran vs. deferred

**Verified GREEN** in a throwaway venv (`pyarrow 24.0.0 + scikit-learn 1.9.0 + numpy 2.5.0`, no compiled `_mlrs`):

| Gate | Result |
|------|--------|
| `pytest tests/test_shims.py tests/test_params.py -q` | **255 passed** |
| `test_init_purity_ast` over the full matrix (incl. UMAP/HDBSCAN) | **32 passed** |
| `estimator_checks` fit-free subset + the never-xfailed guard | **82 passed** (3 checks × 27 estimators + 1 guard) |
| All 15 new classes zero-arg constructible + `get_params` | smoke-test green |
| AST-purity over the 15 new `__init__` bodies | all PURE |

**DEFERRED to CI / UAT** (cannot run in this environment — no maturin/`_mlrs`, per project memory "Python wheel untestable in env"):

- The **fit-based** `test_estimator_checks` checks (everything that calls `fit`/`predict`/`transform` reaches `_mlrs`) — this is the live `check_estimator` run (SHIM-03 by-design). The static fit-free subset above is the maximum verifiable gate here.
- Live oracle/numerical parity for the new shims' fit/predict paths.

The static venv is the same approach 16-00-SUMMARY documented; CI/wheel environments that ship `pyarrow` run the static gates unchanged, and a maturin-built backend runs the full `check_estimator` suite.

## Deviations from Plan

**[Rule 2 — Missing critical coverage] Matrix extended to 32 (not just 27) shims.**
- **Found during:** Task 3 — the derived `_exported_shim_names()` set revealed 5 pre-existing exported shims (IncrementalPCA, EmpiricalCovariance, LedoitWolf, GaussianRandomProjection, SparseRandomProjection) that the original hard-coded `ALL_12` never tested. The plan's acceptance criterion explicitly states the matrix must cover "all 32 estimator shims (17 pre-existing + 15 new)".
- **Fix:** Added `EXPECTED_PARAMS`/`SET_PARAM` rows + the IncrementalPCA `n_components` special-case (mirroring PCA) for all 5, so the derived-set guard (`test_matrix_covers_exports`) passes and the full 32-shim surface is now under the static gate.
- **Files:** test_params.py, test_shims.py, test_estimator_checks.py (the 5 are decomposition/covariance/projection — already shipped; only their test rows were added).
- **Commit:** `0fc3106`

No other deviations — the 15 new shim classes were written exactly to the plan/PATTERNS §6 template and the verified `Py*` `#[new]` signatures.

## Authentication Gates

None.

## Known Stubs

None introduced by this plan. `HDBSCAN.probabilities_` / `outlier_scores_` return `None` until the GLOSH feature-space front-end lands — this is the **already-documented Plan-10 behavior** (the Rust accessors return `Option<Vec<F>>`), not a new stub; the shim faithfully surfaces `None`.

## Acceptance Evidence

- `grep -cE 'class (LinearSVC|LinearSVR|MBSGDClassifier|MBSGDRegressor|GaussianNB|MultinomialNB|BernoulliNB|ComplementNB|CategoricalNB|KernelRidge|KernelDensity|SpectralClustering|SpectralEmbedding)' mlrs/*.py` → **13**.
- `grep -c 'class UMAP' mlrs/manifold.py` → **1**; `grep -c 'class HDBSCAN' mlrs/cluster.py` → **1**.
- All 15 new classes import from `mlrs` without `_mlrs` and construct with ZERO required args.
- Exported estimator-shim count (`issubclass(.., MlrsBase)` over `__all__`) → **32**.
- `EXPECTED_PARAMS` length == 32 == exported shim count (`test_matrix_covers_exports` green).
- The 3 fit-free `estimator_checks` are absent from every xfail map (`test_fit_free_checks_never_xfailed` green) and run green for all 27 listed estimators pre-build.

## For Downstream Plans

- Full pure-Python estimator coverage (32 shims) is now in `crates/mlrs-py/python/mlrs/`. Plan 16-12 (traits.rs deletion / phase wrap-up) inherits a complete Python surface; no further shim work needed.
- The test matrix auto-derives from `mlrs.__all__` — any future shim added to `__init__.__all__` is automatically pulled into `ALL_SHIMS`, the AST-purity gate, and the family/round-trip tests, and will fail `test_matrix_covers_exports` until its `EXPECTED_PARAMS`/`SET_PARAM` rows are added.
- The live `check_estimator` run is the open UAT item: build a backend (`maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml`) and run `pytest tests/test_estimator_checks.py` to exercise the fit-based checks.

## Self-Check: PASSED

- `crates/mlrs-py/python/mlrs/naive_bayes.py` — FOUND
- `crates/mlrs-py/python/mlrs/kernel_ridge.py` — FOUND
- `crates/mlrs-py/python/mlrs/density.py` — FOUND
- `crates/mlrs-py/python/mlrs/manifold.py` — FOUND
- `16-11-SUMMARY.md` — FOUND
- Commit `19292af` (11 family shims) — FOUND
- Commit `175ae09` (Spectral + UMAP/HDBSCAN + registration) — FOUND
- Commit `0fc3106` (test matrix expansion) — FOUND
