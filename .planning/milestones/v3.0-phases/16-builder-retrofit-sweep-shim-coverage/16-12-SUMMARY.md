---
phase: 16-builder-retrofit-sweep-shim-coverage
plan: 12
subsystem: trait-surface-convergence
tags: [typestate, traits-deletion, single-trait-surface, convergence-commit, trybuild, compile-fail, shim-gate, bldr-03, phase-end-gate]
requires:
  - "All 29 estimators migrated to mlrs_algos::typestate (Plans 16-00..16-11, sweep-complete at 16-09)"
  - "Both PyO3 importer files (kernel.rs, naive_bayes.rs + cluster/neighbors/linear) off the legacy surface (16-09)"
  - "typestate.rs mirroring all 9 legacy traits (Plan 16-00)"
  - "trybuild compile_fail harness + ui fixtures (Phase 12)"
provides:
  - "crates/mlrs-algos/src/traits.rs HARD-DELETED — the 9 legacy &mut self traits are gone"
  - "lib.rs with `pub mod traits;` + the `pub use traits::{...}` re-export removed; `pub use error::AlgoError` retained"
  - "mlrs_algos::typestate as the SINGLE trait surface for the whole crate (D-01 end-state)"
  - "compile_fail goldens regenerated for the single-surface diagnostic (predict-before-fit still does NOT compile)"
  - "BLDR-03 COMPLETE — full builder/typestate retrofit + legacy-surface deletion + phase-end gate green"
affects:
  - "BLDR-03 (now COMPLETE — held In Progress through all prior waves, completed by this deletion + verification)"
  - "Phase 16 as a whole (this is the convergence commit; /gsd-verify-work runs next)"
tech-stack:
  added: []
  patterns:
    - "Convergence-by-empty-grep deletion (Pitfall 3): traits.rs is deleted ONLY after a grep for `mlrs_algos::traits` / `crate::traits` across BOTH crates' src (excluding traits.rs) returns EMPTY — the deletion's precondition is that every estimator AND every PyO3 importer is already off the old surface, so the deletion cannot break the cross-crate build."
    - "Doc-link + comment scrub as part of the deletion: after deleting a module, every intra-doc link `[`crate::traits::X`]` becomes a broken rustdoc link and every textual `crate::traits` mention would fail the convergence grep — so the deletion commit also repoints the live link (cluster/mod.rs -> crate::typestate::PredictLabels) and converts the rest to plain code spans / reworded comments."
    - "Single-surface trybuild golden collapse: deleting the duplicate `Transform` trait changes rustc's diagnostic from the fully-qualified `mlrs_algos::typestate::Transform` (needed to disambiguate two same-named traits) to the bare `Transform` (now unambiguous). The .stderr goldens are regenerated via TRYBUILD=overwrite; the VALUE gate (non-compilation naming `Unfit` vs `Fitted`) is unchanged — this is the expected, documented consequence of the convergence, not a regression."
key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/traits.rs   # DELETED
    - crates/mlrs-algos/src/lib.rs
    - crates/mlrs-algos/src/typestate.rs
    - crates/mlrs-algos/src/cluster/mod.rs
    - crates/mlrs-algos/src/cluster/spectral_embedding.rs
    - crates/mlrs-algos/tests/ui/predict_before_fit.stderr
    - crates/mlrs-algos/tests/ui/transform_before_fit.stderr
    - crates/mlrs-py/src/estimators/cluster.rs
    - crates/mlrs-py/src/estimators/kernel.rs
    - crates/mlrs-py/src/estimators/linear.rs
    - crates/mlrs-py/src/estimators/naive_bayes.rs
    - crates/mlrs-py/src/estimators/neighbors.rs
decisions:
  - "traits.rs HARD-DELETED and `pub mod traits;` + the crate-root `pub use traits::{Fit, KNeighbors, PartialFit, Predict, PredictLabels, PredictLogProba, PredictProba, ScoreSamples, Transform};` re-export removed from lib.rs. `pub use error::AlgoError;` is RETAINED (it re-exports from `error`, not `traits` — nothing changed for `mlrs_algos::AlgoError` consumers). Verified beforehand that nothing consumes the trait re-exports via `mlrs_algos::Fit` etc. (grep across all crates + tests returned zero)."
  - "The remaining textual `crate::traits` / `mlrs_algos::traits` occurrences (all doc-links / comments, never live code per the 16-09 sweep) were scrubbed AS PART of the deletion commit, because (a) the `[`crate::traits::X`]` intra-doc links would become broken rustdoc links after deletion, and (b) the plan's verify gate greps the literal strings across both crates and would fail on any residual mention. cluster/mod.rs's live PredictLabels doc-link was repointed to `crate::typestate::PredictLabels`; the two typestate.rs self-referential links were converted to plain `&mut self` code spans; the spectral_embedding.rs comment and the 5 PyO3 'fully off mlrs_algos::traits' comments were reworded to 'legacy trait' / 'legacy-trait' phrasing."
  - "compile_fail goldens regenerated (TRYBUILD=overwrite) — the only change is the FQ-path `mlrs_algos::typestate::Transform` collapsing to the bare `Transform` now that there is one trait of that name. Committed separately as a `test(...)` commit (the deletion is `feat(...)`) so the convergence commit stays a clean single-purpose delete."
  - "BLDR-03 marked COMPLETE in this plan (per the orchestrator directive). It was deliberately held In Progress through Waves 0-11; the full retrofit is finished only now that the legacy surface is deleted AND the phase-end gate is green."
  - "Targeted oracle suite (not blind-full) per RESEARCH disk/time constraints: ran 13 suites covering both pilots (Ridge, MBSGDRegressor), the typestate harness, and one representative per module shape (linear/decomposition/cluster/covariance/projection/density/neighbors/kernel_ridge/naive_bayes) + the HDBSCAN exemplar — every fit/predict/transform/labels/kneighbors/score_samples accessor that the deleted traits backed. umap_test (the UMAP exemplar) was launched but is exceptionally slow in this environment (the 'backend test suite slow' landmine, >25 min under CPU contention); UMAP's typestate surface is independently proven GREEN by the compile_fail trybuild gate (which instantiates Umap<f32, Unfit/Fitted>), and UMAP never used traits.rs, so its full oracle run is a non-gating slow-suite confirmation left to background/CI."
  - "Python static shim suite run in a throwaway venv (pyarrow 24.0.0 + scikit-learn 1.9.0 + numpy 2.5.0, no compiled _mlrs) — the 16-11 precedent — because the base interpreter lacks pyarrow (PEP 668 managed env; mlrs/_io.py imports pyarrow at module load)."
metrics:
  duration: ~47m
  completed: 2026-06-24
  tasks: 2
  files: 12
status: complete
---

# Phase 16 Plan 12: traits.rs hard-deletion — the single-trait-surface convergence Summary

The FINAL plan of Phase 16 — the convergence commit. With all 29 estimators and both PyO3 importer files already on `mlrs_algos::typestate` (sweep completed at 16-09), this plan hard-deletes `crates/mlrs-algos/src/traits.rs` (the 9 legacy `&mut self` traits), removes its `pub mod traits;` declaration and the crate-root `pub use traits::{...}` re-export from `lib.rs`, scrubs the last doc-link/comment mentions of the old path, and runs the phase-end gate. After this commit **`mlrs_algos::typestate` is the ONLY trait surface in the crate** — zero permanent two-surface debt, realizing the Phase-12 D-07 end-state. **BLDR-03 is now COMPLETE.**

## What Was Built

### Task 1 — Pre-deletion grep gate → delete traits.rs + scrub references (convergence commit `d9526ed`)

**Pre-deletion convergence grep (Pitfall 3 precondition):** `grep -rln "mlrs_algos::traits\|crate::traits"` across `crates/mlrs-algos/src` AND `crates/mlrs-py/src` (excluding `src/traits.rs`) returned matches ONLY in doc-comments / plain comments — **zero live code references**. Independently confirmed that nothing consumes the crate-root trait re-exports via `mlrs_algos::{Fit, Predict, ...}` (grep across all crates + tests = empty) and that no `pub use crate::traits::*` re-export exists anywhere besides the explicit list in lib.rs. The deletion was therefore safe (it cannot orphan a name or break either crate).

**The deletion:**
- `git rm crates/mlrs-algos/src/traits.rs` — the 9 legacy traits (`Fit` / `PartialFit` / `Predict` / `Transform` / `PredictLabels` / `KNeighbors` / `ScoreSamples` / `PredictProba` / `PredictLogProba`) are gone.
- `lib.rs`: removed `pub mod traits;` and the whole `pub use traits::{Fit, KNeighbors, PartialFit, Predict, PredictLabels, PredictLogProba, PredictProba, ScoreSamples, Transform};` block; rewrote the module-index doc to point at `[`typestate`]` as the single surface; **kept `pub use error::AlgoError;`** (unaffected — re-exports from `error`, not `traits`).

**Reference scrub (so the post-deletion grep is empty and rustdoc has no broken links):**
- `cluster/mod.rs:5` — the live `[`PredictLabels`](crate::traits::PredictLabels)` intra-doc link repointed to `crate::typestate::PredictLabels`.
- `typestate.rs` — the two self-referential `[`crate::traits`]` / `[`crate::traits::Fit`]` links converted to plain `&mut self` code spans; the module doc updated to past tense ("`traits.rs` was HARD-DELETED in Phase 16 … this is the only trait surface").
- `cluster/spectral_embedding.rs:49` — comment `NO `crate::traits` import` → `NO legacy-trait import`.
- 5 PyO3 comments (`naive_bayes.rs`, `kernel.rs`, `cluster.rs`, `neighbors.rs`, `linear.rs`) reworded from `… off/removed `mlrs_algos::traits`` to `legacy trait` / `legacy-trait` phrasing.

**Verify gate (all green):** post-delete grep across both crates = EMPTY; `test ! -f crates/mlrs-algos/src/traits.rs` passes; `cargo build -p mlrs-algos --features cpu` clean; `cargo build -p mlrs-py --features cpu` clean (only the two pre-existing dead-code warnings on unrelated estimators' `Unfit` fields, out of scope — noted in 16-09).

### Task 2 — Phase-end gate: compile_fail (trybuild) + targeted oracle suite + Python static shim suite

**(1) trybuild compile_fail — GREEN (after golden regen, commit `aee8f9e`).** The first run reported a `.stderr` mismatch: the goldens named the fully-qualified `mlrs_algos::typestate::Transform<f32>` while the actual output named the bare `Transform<f32>`. Diagnosis: before the deletion there were TWO `Transform` traits in resolution space (`traits::Transform` + `typestate::Transform`), so rustc fully-qualified the path to disambiguate; after the deletion there is exactly one, so rustc prints the short unambiguous name. This is the **expected, documented** (harness header Pitfall 5) consequence of the convergence — NOT a regression. Regenerated both goldens via `TRYBUILD=overwrite`; the diff is exactly the FQ-path → short-name collapse (4 lines/file), and both goldens still name `Unfit` / `Fitted` (the value gate). Re-run: `cargo test --features cpu --test compile_fail` → **ui ok (1 passed)** — predict-before-fit still does NOT compile (`E0277`), the BLDR-02 regression guard (T-12-05) holds.

**(2) Targeted oracle suite — GREEN (13 suites).** Per RESEARCH (full `cargo test` is slow / disk-exhausting), ran a targeted set covering both pilots + the typestate harness + one representative per module shape:

| Suite | Result | Shape covered |
|---|---|---|
| typestate_test | 6 passed | the typestate harness itself |
| ridge_test | 6 passed | pilot A — full build-out (Fit+Predict) |
| mbsgd_regressor_test | 5 passed | pilot B — trait-swap (Fit+Predict) |
| linear_regression_test | 7 passed | linear module (Fit+Predict) |
| pca_test | 11 passed | decomposition (Fit+Transform+inverse) |
| kmeans_test | 7 passed | cluster (Fit+PredictLabels) |
| empirical_covariance_test | 7 passed | covariance (Fit) |
| random_projection_test | 10 passed | projection (Fit+Transform) |
| kernel_density_test | 6 passed | density (Fit+ScoreSamples) |
| nearest_neighbors_test | 5 passed | neighbors (Fit+KNeighbors) |
| kernel_ridge_test | 5 passed | kernel_ridge (Shape-A' adopt Fit+Predict) |
| gaussian_nb_test | 7 passed | naive_bayes (Fit+PredictLabels/Proba/LogProba) |
| hdbscan_test | 40 passed | born-with-convention exemplar (cluster) |

Every fit / predict / transform / predict_labels / kneighbors / score_samples / predict_proba / predict_log_proba accessor that the deleted traits backed is exercised and green. (`umap_test`, the UMAP exemplar, was launched but is exceptionally slow here — see Deferred Items; its typestate surface is already proven by the compile_fail gate.)

**(3) Python static shim suite — GREEN (throwaway venv, no compiled `_mlrs`).** Replicated 16-11's venv (pyarrow 24.0.0 + scikit-learn 1.9.0 + numpy 2.5.0) because the base interpreter lacks pyarrow:
- `pytest tests/test_shims.py tests/test_params.py -q` → **255 passed in 1.24s** (the get_params/set_params/clone round-trip + AST-purity matrix over the full 32-shim exported set — identical to 16-11).
- `pytest tests/test_estimator_checks.py -k "<fit-free checks> or fit_free"` → **82 passed, 1278 deselected** — every fit-free sklearn check (`check_no_attributes_set_in_init`, `check_parameters_default_constructible`, `check_get_params_invariance`) across all 32 shims, plus `test_fit_free_checks_never_xfailed`. The 1278 deselected are the fit-based checks (by-design xfail; they reach `_mlrs`) = the SHIM-03 live `check_estimator` path, deferred to UAT.

My commits touched zero Python files, so this green confirms the Rust trait-surface deletion did not perturb the pure-Python shim (as expected — the shim never referenced the Rust traits).

## The Convergence Holds (29/29, single surface)

After this plan: `crates/mlrs-algos/src/traits.rs` does not exist; `pub mod traits;` is gone; the convergence grep across both crates' src is EMPTY; both crates build; predict-before-fit is a compile error; the targeted oracle suite + the Python static shim suite are green. **`mlrs_algos::typestate` is the single trait surface.** Two-surface debt: zero.

## Deviations from Plan

The plan was executed as written. Three plan-anticipated handling details (not scope deviations):

1. **[Rule 1 — Bug] compile_fail golden regeneration.** Task 2's first run failed on a `.stderr` mismatch caused directly by Task 1's deletion (two `Transform` traits → one collapses the FQ path). The harness header explicitly documents this as the regen-the-golden case (Pitfall 5); the value gate is unchanged. Fixed by `TRYBUILD=overwrite` and committed as a separate `test(...)` commit. Files: `tests/ui/predict_before_fit.stderr`, `tests/ui/transform_before_fit.stderr`. Commit `aee8f9e`.
2. **[Rule 3 — Blocking] Doc-link/comment scrub folded into the deletion commit.** The plan's verify greps the literal `crate::traits` / `mlrs_algos::traits` strings across both crates; after deletion those strings survived only in doc-links/comments, which would (a) fail the grep and (b) leave broken rustdoc intra-doc links. Scrubbing them is a precondition for a clean deletion, so it landed in the convergence commit `d9526ed` (not a separate change). No behavior change.
3. **[Rule 3 — Blocking] Throwaway venv for the Python static suite.** The base interpreter has no pyarrow (PEP 668 managed env; `mlrs/_io.py` imports it at module load via `base.py`), so the suite fails to collect. Created a throwaway venv with the 16-11 stack (pyarrow/sklearn/numpy/pytest) and ran the static gate green. pyarrow/scikit-learn/numpy are well-known, prior-wave-used packages — not a package-legitimacy concern.

## Authentication Gates

None.

## Threat Mitigations (from the plan's threat_model)

- **T-16-PITFALL3** (traits.rs deletion vs mlrs-py importers): mitigated by the empty-grep gate ACROSS BOTH CRATES as the deletion precondition. The grep returned zero live references before deletion; the verify re-greps after and would have blocked the commit on any residual. ✓
- **T-16-SEAL** (single trait surface): after deletion `typestate` is the only surface; `State` stays sealed; no third lifecycle state introduced. The deletion removed code only — no new trait/state added. ✓
- **T-16-FFI-DEFER** (live check_estimator): accepted/deferred to UAT — no maturin+pyarrow host. The static (255 + 82 passed) + trybuild (compile_fail green) gates are the maximum verifiable here. ✓

## Known Stubs

None. This plan deletes code and verifies; it adds no stubs.

## Threat Flags

None — a trait-surface deletion plus its verification gate. No new network/auth/file/schema surface.

## Deferred Items (UAT)

- **Live `check_estimator` FFI run (SHIM-03, by-design).** The 1278 fit-based `estimator_checks` cases reach the compiled `_mlrs` extension, which cannot be built here (no maturin/pyarrow host — project memory "Python wheel untestable in env"). UAT step: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml` then `pytest crates/mlrs-py/python/tests/test_estimator_checks.py` to exercise the fit-based checks. The static fit-free subset (82 passed) is the maximum verifiable static gate.
- **Full `umap_test` oracle run.** Launched but exceptionally slow in this environment (the "backend test suite slow" landmine; >25 min under CPU contention from the parallel shim runs) — left running in the background / for CI. UMAP is a born-with-convention exemplar that never used `traits.rs`, and its typestate `Transform`/`Fit` surface is independently proven GREEN by the compile_fail trybuild gate (which instantiates `Umap<f32, Unfit>` and `Umap<f32, Fitted>`). Non-gating for the deletion.

## Verification Environment Note

`cargo build -p mlrs-py --features cpu` is the cross-crate PyO3 gate (the live wheel pytest is untestable here — no maturin/pyarrow). The targeted oracle suites + the mlrs-py build + the trybuild compile_fail gate are the compensating Rust gates; the Python static shim suite (run in the 16-11 throwaway venv) is the compensating Python gate. The live `check_estimator` boundary behavior is unchanged from the pre-deletion shells (this plan touched zero Python files).

## Acceptance Evidence

- Post-delete convergence grep across `crates/mlrs-algos/src` + `crates/mlrs-py/src` (excluding traits.rs) → **EMPTY**.
- `test ! -f crates/mlrs-algos/src/traits.rs` → passes (file deleted); `pub mod traits;` + `pub use traits::{...}` removed from lib.rs; `pub use error::AlgoError;` retained.
- `cargo build -p mlrs-algos --features cpu` → Finished clean.
- `cargo build -p mlrs-py --features cpu` → Finished (2 pre-existing dead-code warnings, out of scope).
- `cargo test --features cpu --test compile_fail` → **ui ok (1 passed)** — predict-before-fit does NOT compile (E0277 naming Unfit/Fitted).
- Targeted oracle suite: 13 suites, all green (typestate 6, ridge 6, mbsgd_regressor 5, linear_regression 7, pca 11, kmeans 7, empirical_covariance 7, random_projection 10, kernel_density 6, nearest_neighbors 5, kernel_ridge 5, gaussian_nb 7, hdbscan 40).
- `pytest tests/test_shims.py tests/test_params.py -q` → **255 passed**.
- `pytest tests/test_estimator_checks.py -k "<fit-free> or fit_free"` → **82 passed, 1278 deselected**; `test_fit_free_checks_never_xfailed` → passed.
- BLDR-03 → COMPLETE.

## Self-Check: PASSED
