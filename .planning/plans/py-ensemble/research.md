# PY-ENSEMBLE Verification Research — Delta Check vs `.planning/plans/RESEARCH.md` (2026-07-16)

**Agent:** Research (Spec-TDD workflow, verification pass). **Date:** 2026-07-17.
**Purpose:** Re-verify every load-bearing claim in `.planning/plans/RESEARCH.md` §5 ("Recommended Feature — Deep Dive", the PY-ENSEMBLE binding plan) against the CURRENT repo state, and determine whether PY-ENSEMBLE is still the correct, safe, unblocked next feature to spec/plan.
**Companion reports (read in full, not reproduced):** `.planning/plans/RESEARCH.md` (full gap survey + PY-ENSEMBLE deep dive), `.planning/plans/RESEARCH-METRICS.md` (metrics-surface deep dive, now landed).
**Evidence labels:** `[VERIFIED: LOCAL <cmd/path>]` · `[VERIFIED: CODEGRAPH <symbol/path>]` · `[UNVERIFIED: …]`

---

## (a) Delta since RESEARCH.md (2026-07-16)

**What changed:**

1. **Metrics surface landed** — commit `0788e17 "Add sklearn metrics surface (mlrs.metrics)"` (2026-07-17), exactly as `.planning/plans/RESEARCH-METRICS.md` planned. It added a wholly new, additive surface: `crates/mlrs-algos/src/metrics/{mod,classification,regression}.rs`, `crates/mlrs-py/src/metrics.rs` (14 `#[pyfunction]` registrations: 11 metrics + 3 `_per_class` variants), `crates/mlrs-py/python/mlrs/metrics.py`, new fixtures `tests/fixtures/metrics_*.npz`, and new tests. `[VERIFIED: LOCAL git show 0788e17 --stat]`
2. **HGB algos churn is STILL live and has progressed, not settled.** `git status` today shows the *same* files RESEARCH.md flagged as risk #3 (`crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`, `crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, all four `hgb_*_seed42.npz` fixtures) still modified and uncommitted — but the working-tree content has moved further since the 2026-07-16 snapshot: a new `gather_hist` helper and sibling-histogram-SUBTRACTION machinery (`gbt_hist_subtract`) were added to the backend prim, `gbt.rs` grew a matching kernel, and `gen_oracle.py`'s HGB fixture generators gained a `rng_offset` parameter with a docstring explicitly describing continued empirical tuning of float-noise tie margins ("the committed value passed cpu but a stale one failed on wgpu at f64... one offset produced a near-tie ~2.4e-5 off at f64 tolerance"). The four committed `hgb_*.npz` fixtures' bytes differ from `HEAD` (md5 mismatch confirmed) even though `git diff --stat` reports unchanged byte counts. `[VERIFIED: LOCAL git status --short; git diff --stat -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py; git diff -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs; md5sum tests/fixtures/hgb_*.npz vs git show HEAD:...]`
3. **The public estimator API surface is untouched by this churn.** `crates/mlrs-algos/src/ensemble/*.rs` (the four `RandomForestClassifier`/`RandomForestRegressor`/`HistGradientBoostingClassifier`/`HistGradientBoostingRegressor` structs, their builders, defaults, and trait impls) show **zero** modifications in `git status` or `git log` since `fb0c9c7` — the churn is confined to `mlrs-backend`'s internal prim implementation, `mlrs-kernels`' device kernel, and the fixture-generation script. `[VERIFIED: LOCAL git status --short | grep ensemble → empty; git log --oneline -- crates/mlrs-algos/src/ensemble/ → only fb0c9c7]`
4. **PY-ENSEMBLE remains completely unstarted.** `ls crates/mlrs-py/src/estimators/` still shows only `cluster.rs, covariance.rs, decomposition.rs, kernel.rs, linear.rs, manifold.rs, mod.rs, naive_bayes.rs, neighbors.rs, projection.rs, spectral.rs` — no `ensemble.rs`. `grep -rniE "randomforest|histgradient|ensemble" crates/mlrs-py --include=*.py --include=*.rs` returns nothing. `[VERIFIED: LOCAL ls; grep]`
5. **`lib.rs` registration state is exactly as RESEARCH.md described**, plus the additive metrics registrations. 32 `m.add_class::<Py...>()?;` calls total (30 non-typestate + `PyUMAP` + `PyHDBSCAN`), matching the "registration 25 → 30" + "Phase-12... " comments RESEARCH.md cited; the doc comment at the top of the `_mlrs` fn still literally says "Register all 12 estimator `#[pyclass]` wrappers" (stale, as RESEARCH.md already flagged — unchanged, still needs correcting whenever ensemble lands). Metrics added a NEW comment block (lines ~270+) registering its 14 free functions via `wrap_pyfunction!` — this is additive and does not touch or renumber the estimator-class registrations. `[VERIFIED: LOCAL grep -c "add_class::<Py" crates/mlrs-py/src/lib.rs → 32; sed -n crates/mlrs-py/src/lib.rs]`
6. **`estimators/mod.rs`'s doc comment ("The 12 `#[pyclass]` estimator wrappers")** and its 10 `pub mod` lines are unchanged — metrics correctly did NOT touch this file, consistent with RESEARCH-METRICS.md's own claim that metrics are free functions, not estimators, and therefore exempt from the estimator-enumerating machinery. `[VERIFIED: LOCAL Read crates/mlrs-py/src/estimators/mod.rs]`
7. **Binding template files are byte-identical in provenance** (last touched pre-ENSEMBLE-01, unaffected by either the ensemble or metrics commits): `crates/mlrs-py/src/estimators/naive_bayes.rs` (1061 lines), `crates/mlrs-py/python/mlrs/naive_bayes.py` (220 lines), `crates/mlrs-py/python/mlrs/base.py` (182 lines), `crates/mlrs-py/src/dispatch.rs` (`any_estimator!`/`any_estimator_typestate!` macros both present, unchanged). `crates/mlrs-py/python/tests/test_oracle_neighbors.py` still exists with the exact structure RESEARCH.md described (`_atol` dtype-branch, `@requires_f64`, fixture replay). `[VERIFIED: LOCAL git log -- <these 4 files> → last commits d9526ed/19292af/1c01eeb, all pre-dating fb0c9c7 and 0788e17; wc -l; Read test_oracle_neighbors.py:1-40]`
8. **The estimator-enumerating gate tests RESEARCH.md named as "must change"** (`test_params.py`, `test_shims.py`, `test_estimator_checks.py`) exist unmodified in `crates/mlrs-py/python/tests/`. RESEARCH-METRICS.md independently confirmed (and this pass re-confirms by directory listing) that metrics correctly bypassed these three files since they are estimator-only gates — no drift to account for. `[VERIFIED: LOCAL ls crates/mlrs-py/python/tests/]`

**Net effect on the PY-ENSEMBLE plan:** the binding-layer plan (§5.1–§5.3, §5.5–§5.7 of RESEARCH.md) is **unaffected and still accurate**. The oracle/fixture plan (§5.4) for **HGB specifically** carries a materially *higher* risk than RESEARCH.md's original MEDIUM rating: the churn is not just "in flight" as a static fact, it is **actively progressing between the two prior research passes** (2026-07-16 → 2026-07-17), with the fixture-generation docstrings themselves describing ongoing empirical probing of float-noise tie margins on top of a not-yet-committed kernel optimization. RF's oracle/fixture plan is fully stable (zero touches to RF prims, kernels, or `rf_*.npz` fixtures in this same window).

---

## (b) Confirmation/correction of every §5 claim (fresh evidence)

### §5.1 Rust core surface

| Claim | Status | Fresh evidence |
|---|---|---|
| Four typestate estimators `Struct<F, S=Unfit>`, consuming `Fit::fit` → `Fitted` sibling | **CONFIRMED** | `[VERIFIED: CODEGRAPH random_forest_classifier.rs:56 pub struct RandomForestClassifier<F, S = Unfit>; fit(self, ...) -> Result<RandomForestClassifier<F, Fitted>, AlgoError> at :366-419]` |
| `RandomForestClassifier::fit` requires `y`, `ingest_labels` → `classes_`/`n_classes_` | **CONFIRMED**, same line range (:366-418 vs RESEARCH.md's :366-418) | `[VERIFIED: CODEGRAPH random_forest_classifier.rs:373-418]` |
| `PredictProba::predict_proba` → n_query×n_classes, rows sum to 1 | **CONFIRMED**, same line (:427) | `[VERIFIED: CODEGRAPH random_forest_classifier.rs:421-440]` |
| `classes()`, `n_classes()`, `model()`; **no `feature_importances_`, no `oob_score_`** | **CONFIRMED** — grep for `feature_importances\|oob_score` across `crates/mlrs-algos/src/ensemble/` returns zero hits | `[VERIFIED: LOCAL grep -n "feature_importances\|oob_score\|PredictLogProba" -r crates/mlrs-algos/src/ensemble/ → no hits]` |
| Builder setters + defaults: `n_estimators=100, max_depth=10, n_bins=32, max_features=Sqrt(clf)/All(reg), min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=true, seed=42` | **CONFIRMED VERBATIM** for both classifier and regressor const blocks | `[VERIFIED: CODEGRAPH random_forest_classifier.rs:42-49 RF_CLF_DEFAULT_*; random_forest_regressor.rs:39-44 RF_REG_DEFAULT_*]` |
| `RandomForestRegressor::fit`/`predict` shape, `model()` only | **CONFIRMED** | `[VERIFIED: CODEGRAPH random_forest_regressor.rs:296-312 Predict::predict]` |
| HGB classifier: `PredictProba` (sigmoid/softmax) + `PredictLabels` (host argmax, ONE metered readback, strict-`>` lowest-index tie-break) | **CONFIRMED VERBATIM**, including the exact tie-break comment | `[VERIFIED: CODEGRAPH hist_gradient_boosting_classifier.rs:313-370]` |
| HGB builder setters/defaults: `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64, l2=0.0, min_samples_leaf=20` | **CONFIRMED VERBATIM** | `[VERIFIED: CODEGRAPH hist_gradient_boosting_regressor.rs:45-50 HGB_REG_DEFAULT_*; classifier const block analogous]` |
| HGB regressor `predict` → raw ensemble scores (baseline + shrunk leaf sums), `model()` only | **CONFIRMED** | `[VERIFIED: CODEGRAPH hist_gradient_boosting_regressor.rs:327-344]` |
| Traits to import: `Fit, Predict, PredictLabels, PredictProba`, **NOT** `PredictLogProba` | **CONFIRMED** — `PredictLogProba` trait exists in `typestate.rs` but has zero impls under `ensemble/` | `[VERIFIED: LOCAL grep PredictLogProba crates/mlrs-algos/src/typestate.rs (trait def only) and crates/mlrs-algos/src/ensemble/ (no impl)]` |

**MaxFeatures enum** (`ensemble/mod.rs:48-71` in RESEARCH.md) — reconfirmed at the same shape: `Sqrt | Log2 | All | Value(usize)`, `resolve(n_features)` method. `[VERIFIED: LOCAL Read crates/mlrs-algos/src/ensemble/mod.rs:1-75]`. The module doc's "Deviations from sklearn" block (histogram-binned trees, `max_depth` bounded `1..=16`, deterministic tie-break, no early stopping) is also unchanged verbatim.

### §5.2 Files to create/modify

All still accurate as a plan of record: `estimators/ensemble.rs` (new), `estimators/mod.rs` (add `pub mod ensemble;` — currently exactly 10 submodules, confirmed), `lib.rs` (add 4 `use`/`add_class` — currently 32 add_class calls, would become 36; the metrics work's new `wrap_pyfunction!` block does not conflict), `python/mlrs/ensemble.py` (new), `python/mlrs/__init__.py` (add import + `__all__`), `python/tests/test_oracle_ensemble.py` (new), `test_params.py`/`test_shims.py`/`test_estimator_checks.py` (must extend — confirmed still estimator-keyed and untouched by metrics). `[VERIFIED: LOCAL Read crates/mlrs-py/src/estimators/mod.rs; grep -c add_class crates/mlrs-py/src/lib.rs]`

`any_estimator_typestate!` macro trap (use it, NOT `any_estimator!`, because ensemble cores default `S=Unfit`) — **reconfirmed**, macro still present at `dispatch.rs` with the identical doc-comment warning about "the WRONG monomorphization" cited in RESEARCH.md. `[VERIFIED: LOCAL Read crates/mlrs-py/src/dispatch.rs:90-190]`

### §5.3 Binding pattern (GIL release, sanctioned lock, f64 guard, build-before-upload, egress, dtype dispatch)

No changes to `dispatch.rs`, `naive_bayes.rs` (template), or `lib.rs`'s `lock_pool`/`capability` machinery since RESEARCH.md was written — all five contracts remain exactly as documented, with the naive_bayes.rs template still the correct one to mirror for classifiers (1061 lines, last touched at `1c01eeb`/`19292af`/`d9526ed`, all pre-ensemble). `[VERIFIED: LOCAL git log -- crates/mlrs-py/src/estimators/naive_bayes.rs crates/mlrs-py/src/dispatch.rs]`

### §5.4 Oracle/fixture convention

- Fixture existence (8 files at `tests/fixtures/{rf,hgb}_{cls,reg}_{f32,f64}_seed42.npz`) — **CONFIRMED**, all 8 present.
- RF fixture keys/tiers (deterministic `bootstrap=false, max_features=All, depth=12, n_estimators=2`; statistical `n_estimators=64, depth=8`; `ACC_MARGIN=0.05`) — **CONFIRMED**, `crates/mlrs-algos/tests/random_forest_classifier_test.rs` is untouched since `fb0c9c7` (`git diff` empty), so every key name and tier assertion RESEARCH.md cited is still exactly as documented. **RF fixtures are stable and safe to bind against today.**
- HGB deterministic tier (`n_bins=255` override required, `max_leaf_nodes=None` equivalence argument, sklearn det kwargs) — **STRUCTURALLY CONFIRMED** (the Rust test file `hist_gradient_boosting_classifier_test.rs` still builds the deterministic-tier classifier with `.n_bins(255)` and reads `proba_key`/`pred_key`/`y_key`-parameterized fixture fields plus `stat_acc_test`), but the underlying fixture BYTES and the internal split-tie-breaking algorithm they were generated against are **actively being retuned** (see delta §a.2). This is a strengthening, not a weakening, of RESEARCH.md's risk #3 — the correct fixture keys/shape are stable, but their numeric content is not yet final. `[VERIFIED: LOCAL Read crates/mlrs-algos/tests/hist_gradient_boosting_classifier_test.rs:100-190; git diff --stat scripts/gen_oracle.py + tests/fixtures/hgb_*.npz]`
- Regen path / venv instructions — unchanged, still `numpy scipy scikit-learn` in a PEP-668 venv; sklearn version used to produce the committed fixtures remains unstamped in-repo (**Q3 still open**, see below). `[VERIFIED: LOCAL scripts/gen_oracle.py:15-17]`
- Capability gate (`skip_f64_with_log` Rust-side, `@requires_f64` Python-side) — unchanged.

### §5.5–§5.6 Builder/typestate convention, two-tier stochastic gate

Unchanged and reconfirmed: v3 typestate builder pattern (`Struct::<F>::builder()....build::<F>()` returning `Result`), RF's bootstrap-stochastic two-tier gate (deterministic `bootstrap=false` + statistical accuracy band), HGB's RNG-free deterministic-exact tier plus a statistical-defaults band. `[VERIFIED: CODEGRAPH ensemble/mod.rs; LOCAL hist_gradient_boosting_classifier_test.rs]`

### §5.7–§5.8 Validation commands, dependency versions

No changes to `Cargo.toml`/`Cargo.lock` pins since RESEARCH.md (`pyo3 0.28.3`, `arrow 59.0.0`, `cubecl 0.10.0`, `abi3-py312`) were touched by either the ensemble or metrics work (metrics is a pure Rust/Python addition with no new external dependency, confirmed by RESEARCH-METRICS.md and by `git show 0788e17 --stat` showing no `Cargo.toml`/`Cargo.lock` diff). Validation commands (`cargo test -p mlrs-algos --features {cpu,wgpu}`, `cargo test -p mlrs-py --features cpu`, `maturin develop ... && pytest crates/mlrs-py/python/tests/`) remain accurate; no `justfile`/`Makefile`/CI exists. `[VERIFIED: LOCAL git show 0788e17 --stat | grep -i cargo → no hits]`

---

## (c) Final verdict

**PY-ENSEMBLE is still the correct next feature to spec and plan — but the plan must be explicitly split (RF now, HGB gated) rather than proceeding as one undivided unit, because the HGB half's oracle-fixture ground truth is demonstrably still moving in the working tree.**

Specifically:

- **RandomForestClassifier / RandomForestRegressor: unblocked, safe to spec and plan immediately.** Rust core, builder defaults, trait surface, oracle test, and the two RF `.npz` fixtures are all untouched since `fb0c9c7` (zero diff, verified by `git log`/`git diff` on the exact files). Binding templates (`naive_bayes.rs`, `dispatch.rs`, `base.py`, `naive_bayes.py`, `test_oracle_neighbors.py`) are stable and unaffected by either subsequent commit. Nothing about the metrics landing (`0788e17`) touched anything RF depends on.
- **HistGradientBoostingClassifier / HistGradientBoostingRegressor: the algos layer is still churning as of this research pass**, more concretely than RESEARCH.md's original "in flight" framing — a sibling-histogram-subtraction kernel change is mid-refinement with an explicitly tunable, still-being-probed RNG offset for float-noise tie margins, and the four `hgb_*.npz` fixtures on disk already differ from the last commit (`fb0c9c7`) without a corresponding commit capturing the new state. Binding an HGB Python oracle test against today's uncommitted fixture bytes risks an immediate break the moment this churn lands (fixture regen would silently invalidate any pinned `atol`/`n_bins=255` deterministic-tier assertions written against the current working-tree `.npz` files). **This confirms and sharpens RESEARCH.md's Q4 recommendation ("bind RF first")** — it should now be treated as a hard sequencing constraint, not just a preference: land/commit the HGB algos churn (or explicitly snapshot+freeze the fixtures the Python oracle will bind against) before finalizing HGB Python oracle tolerances.

**Recommended scope narrowing for the SPEC/PLAN:** structure PY-ENSEMBLE as two sequenced units —
1. **Unit 1 (do now):** `RandomForestClassifier` + `RandomForestRegressor` Python bindings, full oracle test, full estimator-gate-test updates. No blocking risk.
2. **Unit 2 (gate on HGB algos settling):** `HistGradientBoostingClassifier` + `HistGradientBoostingRegressor` Python bindings. Before finalizing this unit's oracle tolerances, either (a) confirm `git status` is clean on `hist_gradient_boosting.rs`/`gbt.rs`/`gen_oracle.py`/`hgb_*.npz` (i.e., the sibling-subtraction work has been committed and fixtures are final), or (b) if the Planner/user wants to proceed in parallel, explicitly bind against a git-pinned commit of the current fixtures and flag that a fixture regen is a known, planned follow-up before ship.

This does not block writing the ensemble.rs module structurally for all four estimators (the `#[pyclass]` wrapper shape, builder setter mapping, and `MaxFeatures` parsing are identical in mechanism for RF and HGB) — only the HGB **oracle-test tolerance pinning** needs to wait on settled fixtures. A single PLAN could implement all four wrappers together and gate only the HGB oracle-test-finalization task on the churn resolving, if the Planner prefers not to split into two units. Either sequencing is valid; the research constraint is specifically "do not pin HGB Python oracle numbers against fixtures known to be mid-regen."

---

## Open questions re-checked (from RESEARCH.md §6)

- **Q1 (`predict_log_proba` scope):** still unresolved — Rust cores still have no `PredictLogProba` impl for any ensemble estimator (reconfirmed by grep). Owner: Planner + user, unchanged.
- **Q2 (`feature_importances_`/`oob_score_` scope):** still unresolved and still absent from the Rust core (reconfirmed by grep across `ensemble/`). Recommend defer, as RESEARCH.md did.
- **Q3 (fixture sklearn version):** still unstamped in-repo for RF fixtures. For HGB fixtures this question is now entangled with the active churn — the version question is now secondary to "which fixture generation (rng_offset) is final." Owner: Planner, unchanged in nature but higher urgency for HGB.
- **Q4 (RF-before-HGB sequencing):** **now upgraded from a recommendation to a required constraint** per the verdict above — the churn evidence is stronger today than on 2026-07-16.
- **New Q6 (this pass):** Should the SPEC explicitly call out "HGB oracle tolerances must be finalized against a committed, non-dirty fixture state" as an acceptance-criterion precondition, or should the Planner accept binding against the current dirty working-tree fixtures with a documented follow-up task? Owner: Planner + user (blocks HGB-half finalization only, not RF).

---

## Sources

- `.planning/plans/RESEARCH.md` (full text read) — original gap survey + PY-ENSEMBLE deep dive, 2026-07-16.
- `.planning/plans/RESEARCH-METRICS.md` (full text read) — metrics-surface deep dive, 2026-07-16, now landed as `0788e17`.
- `[VERIFIED: LOCAL]` — `git status --short`, `git log --oneline -5`, `git diff --stat`, `git show fb0c9c7 --stat`, `git show 0788e17 --stat`, `git diff --stat -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py`, `git diff -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`, `md5sum tests/fixtures/hgb_*.npz` vs `git show HEAD:tests/fixtures/hgb_cls_f32_seed42.npz | md5sum`, `git status --short | grep -i ensemble` (empty), `git log --oneline -- crates/mlrs-algos/src/ensemble/`, `git log -3 --oneline -- crates/mlrs-py/src/estimators/naive_bayes.rs crates/mlrs-py/python/mlrs/naive_bayes.py crates/mlrs-py/python/mlrs/base.py crates/mlrs-py/src/dispatch.rs`, `ls crates/mlrs-py/src/estimators/`, `ls crates/mlrs-py/python/mlrs/ crates/mlrs-py/python/tests/ crates/mlrs-py/tests/`, `grep -rniE "randomforest|histgradient|ensemble" crates/mlrs-py`, `grep -n "feature_importances|oob_score|PredictLogProba" -r crates/mlrs-algos/src/ensemble/ crates/mlrs-algos/src/typestate.rs`, `grep -c "add_class::<Py" crates/mlrs-py/src/lib.rs`, Read of `crates/mlrs-py/src/lib.rs`, `crates/mlrs-py/src/estimators/mod.rs`, `crates/mlrs-py/src/dispatch.rs:90-190`, `crates/mlrs-algos/src/ensemble/mod.rs`, `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:1-60`, `crates/mlrs-algos/tests/hist_gradient_boosting_classifier_test.rs:100-190`, `crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-40`.
- `[VERIFIED: CODEGRAPH]` — `codegraph_explore` query "RandomForestClassifier RandomForestRegressor HistGradientBoostingClassifier HistGradientBoostingRegressor builder setters defaults classes n_classes model predict_proba predict_labels feature_importances oob_score", returning verbatim source of `crates/mlrs-algos/src/ensemble/{random_forest_classifier,random_forest_regressor,hist_gradient_boosting_classifier,hist_gradient_boosting_regressor}.rs` plus blast-radius (no covering Python tests exist for any ensemble symbol — consistent with the PY-ENSEMBLE gap being real).

---

## Confidence Assessment

- **HIGH:** Every §5.1–§5.3, §5.5–§5.8 claim in RESEARCH.md re-verified unchanged. RF algos/fixtures/tests fully stable (zero diff since `fb0c9c7`). PY-ENSEMBLE gap still fully open (zero Python surface). Metrics landing is additive and does not touch anything the ensemble binding plan depends on. Absence of `feature_importances_`/`oob_score_`/`PredictLogProba` reconfirmed by direct grep.
- **MEDIUM:** Exact current numeric content of the four `hgb_*.npz` fixtures (confirmed changed vs `HEAD` but not independently re-validated against a fresh sklearn run in this pass — that was out of scope for a verification-only research task).
- **LOW/UNVERIFIED:** Whether the sibling-histogram-subtraction work will be committed as-is, revised further, or reverted (no visibility into author intent beyond the working-tree diff and docstrings) — this is the key unresolved item gating HGB oracle-tolerance finalization. Exact sklearn version used for either RF or HGB fixtures remains unstamped in-repo (Q3, carried over unchanged from RESEARCH.md).
