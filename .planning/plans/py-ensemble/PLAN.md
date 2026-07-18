---
plan_document: TDD implementation plan
phase: py-ensemble
source_spec: .planning/plans/py-ensemble/SPEC.md
source_research: .planning/plans/py-ensemble/research.md, .planning/plans/RESEARCH.md
generated_at: 2026-07-17
task_count: 25
---

# PY-ENSEMBLE — TDD Implementation Plan

Plans from `SPEC.md` (9 spec IDs: `PY-ENS-01..05`, `RF-IMP-01/02`,
`RF-OOB-01/02`). Every task below cites verified evidence (CodeGraph / Read /
Bash) — no invented path or symbol. `crates/mlrs-kernels/src/tree.rs` and
`crates/mlrs-backend/src/prims/random_forest.rs`, flagged `[UNVERIFIED]` in
SPEC.md §3/§9, were read in full by this Planner (kernel bodies for
`rf_hist_cum`, `rf_node_total`, `rf_node_max`, `rf_split_scores_class/reg`,
`rf_best_split`, `rf_predict_leaf`, `rf_vote_class`, `rf_mean_reg`; the prim's
`RfModel`, `RfParams`, `bootstrap_weights`, `rf_fit_impl` lines 1-870) — the
mechanism for RF-IMP-01/RF-OOB-01 below is now concrete, not a sketch.

This planner did **not** invoke, import, or depend on any GSD skill,
command, workflow, or agent, per the mission's explicit instruction.

## Resolved planning decisions (Planner-owned open questions)

**Q-tree.rs (SPEC §3/§9 risk 4 — the single largest unresolved-evidence risk):**
Read in full via CodeGraph (`crates/mlrs-kernels/src/tree.rs:204-512`) and
`crates/mlrs-backend/src/prims/random_forest.rs:1-870`. Findings:

- `rf_split_scores_class`/`rf_split_scores_reg` (K5, `tree.rs:296-387`) write
  a **gini/variance-reduction PROXY** `sql/nl + sqr/nr` (classifier) or
  `syl²/nl + syr²/nr` (regressor) per candidate split — `[VERIFIED: LOCAL
  tree.rs:317-347,364-386]`. Algebraically this proxy IS a true sklearn
  impurity-decrease building block: for a node with weighted total `tot` and
  winning proxy score `best` (computed inside `rf_best_split`, `tree.rs:441-451`,
  currently a **local variable, never written to any output array** —
  `[VERIFIED: LOCAL tree.rs:428-451]`), the sklearn-equivalent per-node
  weighted impurity decrease is:
  - **Classifier:** `decrease = best − (Σ_c tc²) / tot`, where `Σ_c tc²` is
    the node's OWN class-count sum-of-squares (parent impurity numerator) —
    **not currently computed anywhere** (`rf_node_max`, `tree.rs:262-286`,
    computes the running MAX, not the sum-of-squares; no existing kernel
    reduction produces `Σ_c tc²`).
  - **Regressor:** `decrease = best − (syt²) / tot`, where `syt` is the
    node's total weighted `Σy` — **already available** as
    `hist[hbase+1]` inside `rf_best_split` (`tree.rs:491`, the same value the
    kernel already reads for the leaf-mean branch) — **no new kernel needed
    for the regressor path.**
  (Derivation: weighted-Gini/MSE decrease `= tot·impurity_parent −
  nl·impurity_left − nr·impurity_right` expands, using `nl+nr=tot` and the
  existing `sql`/`sqr`/`syl`/`syr` quantities, to exactly `score −
  parent_sumsq/tot` in both modes — verified algebraically against the
  standard sklearn Gini/MSE-decrease formula.)
- **Mechanism decision (extend `rf_best_split`'s output arrays — NOT a
  separate reduction pass):** add ONE new device kernel `rf_node_sqsum`
  (classifier-only, mirrors `rf_node_max`'s exact per-`(tree_in_chunk,node)`
  running-reduction shape at `tree.rs:262-286`, replacing the running-max
  `if v > mx` with a running-sum `sq += v*v`) as a new K4.5 launched
  alongside K3/K4 in `rf_fit_impl`'s per-level/per-chunk loop
  (`random_forest.rs:651-668`), with a transient `nsq_h` buffer released the
  same way as `ntot_h`/`nmax_h` (`random_forest.rs:754-757`). Extend
  `rf_best_split` (K6) with ONE new output parameter `node_decrease: &mut
  Array<F>` (shape `t * total_nodes`, parallel to `is_leaf`/`leaf_dist`):
  write `0` on leaf nodes, `best − node_sq[tid]/tot` (classifier, reading the
  new `nsq_h` input) or `best − hist[hbase+1]²/tot` (regressor, no new
  input) on split nodes. `rf_fit_impl` allocates ONE new persistent model
  array `node_decrease` (`t * total_nodes * size_of::<F>()`, same lifetime as
  `is_leaf`) and passes it to K6.
- **Reduction to `feature_importances_`:** computed EAGERLY inside
  `rf_fit_impl`, once, right after the level loop (before `Ok(...)` returns)
  — ONE additional host readback of THREE already-persistent device arrays
  (`split_feature`, `is_leaf`, the new `node_decrease` — all `t *
  total_nodes` elements, e.g. 100 trees × 2047 nodes ≈ 205K elements per
  array at `depth=10`, a few MB, one-time, non-blocking since it happens
  after the launch-only level loop completes, not inside it — this is the
  SAME "one readback after launch-only compute" pattern already used for
  `x_host`/edges at `random_forest.rs:461-463`, so it does not reintroduce
  the per-iteration host-sync problem the module doc (`random_forest.rs:1-13`)
  guards against). Host reduction: for each node index `i` where
  `is_leaf[i]==0`, `importances[split_feature[i]] += node_decrease[i]`; then
  normalize the length-`n_features` vector to sum to 1 (guard: if the total
  sum is `0` — e.g. an all-leaf-forest degenerate case — return an all-zero
  vector rather than dividing by zero; **resolves SPEC §4.1's degenerate-case
  TBD** without needing to prove whether early-stop-before-`max_depth` is
  reachable — the guard is correct regardless).
- **`rf_fit_impl` return type (resolves SPEC §4.2's TBD):** adopt the
  SPEC-sketched `RfFitOutcome<F> { model: RfModel<F>, feature_importances:
  Vec<F>, oob_score: Option<F> }` verbatim; `RfModel<F>` itself is UNCHANGED
  except for gaining the new `node_decrease: DeviceArray<ActiveRuntime, F>`
  field (kept device-resident on `RfModel` too, in case a future consumer
  wants it, mirroring the existing `is_leaf`/`leaf_dist` fields — but the
  Python-facing `feature_importances_` accessor reads the already-reduced,
  already-normalized host `Vec<F>` on `RfFitOutcome`/the algos `Fitted`
  struct, NOT a lazy re-reduction, per SPEC §4.1's plain `&[F]`-returning,
  no-pool-argument accessor shape).

**Q-oob (SPEC §3/§9 risk — bootstrap-weight persist-vs-rederive):**
**Rederive**, not persist (SPEC's own recommended option). Evidence: a fresh
`SplitMix64::new(params.seed)` fed to `bootstrap_weights::<F>` reproduces
`w_host` byte-for-byte because it is the FIRST call on a freshly seeded
stream in both the original derivation (`random_forest.rs:467-468`) and the
rederivation — no other RNG draw precedes it (`sample_features` only runs
inside the level loop, strictly after `bootstrap_weights` per the fixed
consumption-order doc comment, `random_forest.rs:11,376-378`). Rederiving is
a **host-only, no-device-sync** `O(t·n)` loop (`[VERIFIED: LOCAL
random_forest.rs:379-397]`), cheaper than retaining an extra `t·n·sizeof(F)`
device buffer for the (common, opt-in-false) case. **Mechanism:** inside
`rf_fit_impl`, gated behind `if params.oob_score { ... }` (so the
non-opted-in common case pays zero extra cost, matching SPEC §5's stated
intent): rederive `w_host` via a second `SplitMix64::new(params.seed)` +
`bootstrap_weights::<F>` call; launch the EXISTING `rf_predict_leaf` kernel
(`tree.rs:636-670`, already used by the predict path, confirmed via
CodeGraph) on the STILL-IN-SCOPE `x` (the function's own `&DeviceArray`
parameter — never dropped, only `x_host` is dropped at
`random_forest.rs:463`) against the just-built model arrays (still local
variables before being moved into `RfModel`), producing `leaf_ids: t × n`
device u32; readback `leaf_ids`, `leaf_dist` (already a local device array,
one-time `to_host`), and `y` (readback the target device array in scope as
`RfTarget::Class(y_dev,_)`/`RfTarget::Reg(y_dev)` — `y_idx` are documented as
DENSE class indices for the classifier per `random_forest.rs:150`); host-side
aggregate, per training row `i`, the mean leaf value/distribution over ONLY
the trees `t` where `w_host[t*n+i] == 0` (out-of-bag); classifier: accuracy
of the argmax vs. the dense class index; regressor: R² vs. `y`. This reuses
existing predict-path kernels exactly as SPEC §3 instructs ("the forest's own
predict-path traversal logic (reused, not reimplemented)").

**Q-zero-oob (SPEC §4.1/RF-OOB-01 unresolved question — signal channel):**
**`log::warn!`** (SPEC's own recommended option (b)), confirmed available:
the `log` crate is a workspace dependency (`Cargo.toml:55`,
`[VERIFIED: LOCAL]`) already used from `mlrs-algos`
(`crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs`,
`[VERIFIED: LOCAL grep log::warn! crates/]`) and `mlrs-backend`
(`capability.rs`, `pool.rs`, `reduce.rs`). A training row with zero OOB trees
is silently excluded from the `oob_score_` aggregation (sample count `0`
denominator skipped) and a single `log::warn!("random_forest: N training
row(s) had zero out-of-bag trees and were excluded from oob_score_ (increase
n_estimators or bootstrap variance)")` is emitted once per `fit()` call (not
once per row) if `N > 0`.

**Q-BuildError-location:** SPEC §3 attributes `validate_forest_hyperparams`
to `ensemble/mod.rs`; **this Planner's fresh read of `ensemble/mod.rs`
(72 lines total) shows it does NOT live there** — it is `pub(crate) fn
validate_forest_hyperparams` defined at the bottom of
`crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:267+`
(`[VERIFIED: LOCAL — full-file read]`), imported by the regressor builder.
**Correction, not a blocker:** TASK-05 adds the `oob_score`/`bootstrap`
cross-check as a small INLINE check in each builder's own `build::<F>()`
(classifier AND regressor), immediately after the existing
`validate_forest_hyperparams(...)?` call, rather than threading a new
parameter through the shared helper — avoids widening
`validate_forest_hyperparams`'s signature (and hence every one of its call
sites) for a check only RF needs. `BuildError::OobRequiresBootstrap {
estimator: &'static str }` is a new variant on the existing `BuildError` enum
in `crates/mlrs-algos/src/error.rs` (`[VERIFIED: LOCAL error.rs:400+ — enum
shape, `#[error(...)]` + named-field variant convention confirmed from
`InvalidAlpha`/`InvalidL1Ratio`/`InvalidEps`]`).

**Q-feature-importances-tolerance (SPEC §5 RF-IMP-01 unresolved question):**
Two-tier, reusing the ALREADY-EXISTING RF fixtures rather than inventing a
new fixture family: (1) **exact tier** — extend the EXISTING, already-stable
`gen_rf_classifier`/`gen_rf_regressor` generators in `scripts/gen_oracle.py`
to ALSO compute `ref_feature_importances` via
`sklearn.ensemble.{RandomForestClassifier,RandomForestRegressor}
.feature_importances_` on the SAME deterministic-tier
(`bootstrap=False, max_features=all`) data already proven to build
structurally-identical trees (`random_forest_classifier_test.rs:36-192`,
`[VERIFIED: LOCAL]`), and ADD this one new array key to the FOUR already-committed
`rf_{cls,reg}_{f32,f64}_seed42.npz` files (purely additive — `np.savez` with
one more named array does not disturb existing consumers reading the
pre-existing keys) — originally proposed at `≤1e-5`, **[RESOLVED at TASK-02
Green time, 2026-07-18: revised to `atol=0.05`]** after discovering sklearn's
own splitter breaks near-tied splits with seed-independent internal state
(so sklearn's own deterministic-tier trees aren't bit-identical to each
other, even though mlrs's are) — see TASK-02's Objective for the full
root-cause and SPEC.md spec_revision 2; (2) **qualitative tier** — a NEW
small hand-built synthetic dataset (one dominant, perfectly-separating
feature + noise features) asserted via a ratio/ranking check
(`importances[dominant] > importances[noise]`), not an exact match, exactly
per SPEC's own recommendation.

**Q-conditional-attribute test machinery (SPEC RF-OOB-02 unresolved
question):** Read `crates/mlrs-py/python/tests/test_shims.py:178-206`
(`[VERIFIED: LOCAL]`): `test_fitted_attr_raises_before_fit` is a flat
`@pytest.mark.parametrize("name,attr", [...])` list asserting
`NotFittedError` — it has **no support** for a THIRD state ("attribute
absent entirely, `AttributeError`, even after fit, when a constructor flag
is `False`"). **Resolution:** `feature_importances_` (always present once
fitted) IS added to this generic list for both RF estimators.
`oob_score_` is **NOT** added to this generic list (it would assert the
wrong exception under the default `oob_score=False` construction); instead
TASK-16 adds a DEDICATED test (in `test_shims.py`, new function, not
`test_fitted_attr_raises_before_fit`) asserting BOTH: (a)
`RandomForestClassifier(oob_score=True).oob_score_` before `fit` raises
`NotFittedError` (standard), and (b)
`RandomForestRegressor(oob_score=False).fit(X,y).oob_score_` raises
`AttributeError` (sklearn-parity, non-standard) — resolving SPEC's flagged
"first mlrs estimator with a conditionally-present fitted attribute" gap by
extending the test FILE with a new test function rather than the shared
parametrized list.

**Q-HGB-fixture-freshness (locked decision, re-verified at plan time):**
`git status --short` (run fresh by this Planner, 2026-07-17) confirms
`crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`,
`crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and all four
`tests/fixtures/hgb_{cls,reg}_{f32,f64}_seed42.npz` are **still modified,
uncommitted** — identical to `research.md`'s 2026-07-17 finding, unchanged
since. TASK-17 is the explicit gate checkpoint; TASK-24 (HGB oracle-tolerance
finalization) is hard-blocked by it as of THIS plan's writing and MUST
re-verify at its own Green time (state may have changed by execution time).

**Q-scope (reconfirmed, no change):** `predict_log_proba` omitted (locked);
`oob_decision_function_`/`oob_prediction_` omitted (locked, narrow reading);
`sample_weight` omitted from all four `fit()` signatures (locked, matches
the 30-signature precedent SPEC §2 already cites); HGB gets no
`feature_importances_`/`oob_score_` (locked, sklearn API-shape
non-applicability).

## Binding-layer template (verified fresh, underlies TASK-08/09/18/19)

`crates/mlrs-py/src/estimators/naive_bayes.rs:365-418` (`PyGaussianNB::fit`,
`[VERIFIED: LOCAL — full read]`) is the exact typestate-fit-with-y template:
`capsule_to_array` (x, y) → `float_dtype` → snapshot ctor fields out of the
`Unfit` arm → `py.detach(|| { lock_pool(); match dt { F32 => { validated_f32
×2, builder().…build::<f32>().map_err(build_err_to_py)?,
TypestateFit::fit(est,&mut pool,&xd,Some(&yd),(rows,cols)).map_err(algo_err_to_py)?
}, F64 => { guard_f64()?; … } } })? ; self.inner = fitted;`. The
`any_estimator_typestate!` macro (`crates/mlrs-py/src/dispatch.rs:158-185`,
`[VERIFIED: LOCAL — full read]`) is confirmed the CORRECT macro (spells
`<F, mlrs_algos::typestate::Fitted>` explicitly in both fitted arms) — using
plain `any_estimator!` (`dispatch.rs:95-119`) would resolve
`RandomForestClassifier<f32, Unfit>` in the `F32` arm (WRONG, SPEC risk 1).
`estimators/mod.rs` currently lists exactly 10 `pub mod` lines with NO
`ensemble` (`[VERIFIED: LOCAL — full file read]`); `lib.rs` currently
registers 32 `add_class::<Py...>()?` calls and carries the stale "12
estimator"/"30" doc comments at `lib.rs:65,178,201` (`[VERIFIED: LOCAL grep]`).
`test_shims.py::ALL_SHIMS` (`test_shims.py:28-42`) is **auto-derived** from
`mlrs.__all__` via `issubclass(obj, MlrsBase)` (`[VERIFIED: LOCAL — full
read]`) — adding the four new names to `__init__.py`'s `__all__` (TASK-13/23)
automatically extends `test_all_shims_importable` /
`test_fit_returns_self_signature` / `test_output_type_param_present` with
**no manual edit needed** in those three tests. `test_params.py::EXPECTED_PARAMS`
(`test_params.py:29+`) and `test_estimator_checks.py::_estimators()`
(`test_estimator_checks.py:33-64`) are, by contrast, **manual** dicts/lists
(`[VERIFIED: LOCAL — full read of both]`) — TASK-16/25 add entries there by
hand.

## Fixture conventions (binds TASK-02/03/06/07 to the existing RF fixtures)

No NEW fixture FILES for RF-IMP-01/RF-OOB-01 — both extend the four
already-committed, already-stable `tests/fixtures/rf_{cls,reg}_{f32,f64}_seed42.npz`
in place (new `np.savez` keys only): `ref_feature_importances` (deterministic
tier, TASK-02/03) and `ref_oob_score` + the `oob_score=True` constructor
kwargs used to produce it (statistical tier, TASK-06/07). PY-ENS-01..04's own
Python-oracle replay (TASK-14/15/24) is a THIRD consumer of these same
files, no regeneration.

## Execution waves (dependency order)

```text
Wave 1 (RF-IMP-01 foundation, sequential — same fn rf_fit_impl):
  TASK-01 -> TASK-02 -> TASK-03
Wave 2 (RF-OOB-01, sequential — same fn rf_fit_impl; functionally depends
  ONLY on TASK-01, not on TASK-02/03 — TASK-01 is what introduces
  `RfFitOutcome`/`rf_fit_impl`'s new return shape that TASK-04 extends;
  TASK-02/03 add the SEPARATE `feature_importances()` algos-layer accessor
  on a DIFFERENT struct field and are not a prerequisite for TASK-04's
  `oob_score` work — Plan-Check Pass-1 Issue 3 clarification, reconciled
  with TASK-04's own "Depends on: TASK-01" field below):
  TASK-04 -> TASK-05 -> TASK-06 -> TASK-07
Wave 3 (PY-ENS-01/02 Rust binding, sequential — same file ensemble.rs):
  TASK-08 -> TASK-09
Wave 4a: TASK-10 (lib.rs registration, RF)      [depends on TASK-09]
Wave 4b (Python shim, sequential — same file ensemble.py; PARALLEL with 4a):
  TASK-11 -> TASK-12                             [depends on TASK-09]
Wave 5: TASK-13 (__init__.py wiring, RF)         [depends on TASK-10, TASK-12]
Wave 6 (Python oracle replay, sequential — same file test_oracle_ensemble.py):
  TASK-14 -> TASK-15                             [depends on TASK-13]
Wave 7: TASK-16 (PY-ENS-05 gate tests, RF)       [depends on TASK-15]
Wave 8: TASK-17 (HGB freshness gate checkpoint)  [no code dependency; PARALLEL with Waves 1-7]
Wave 9 (PY-ENS-03/04 Rust binding, sequential — same file ensemble.rs; depends on TASK-09 file state):
  TASK-18 -> TASK-19
Wave 10a: TASK-20 (lib.rs registration, HGB)     [depends on TASK-19]
Wave 10b (Python shim, sequential; PARALLEL with 10a):
  TASK-21 -> TASK-22                             [depends on TASK-19]
Wave 11: TASK-23 (__init__.py wiring, HGB)       [depends on TASK-20, TASK-22]
Wave 12: TASK-24 (HGB oracle replay, GATED on TASK-17) [depends on TASK-23, TASK-17]
Wave 13: TASK-25 (PY-ENS-05 gate tests, HGB)     [depends on TASK-24]
```

TASK-17 has no code dependency and can run at any point (it only inspects
`git status`); it is placed early enough that its finding is available
before TASK-24 needs it. Waves 1-7 (RF) and Wave 8 (the HGB gate check) are
parallel-eligible with each other (disjoint files: `random_forest*.rs`
vs. nothing/`git status`). Waves 9+ (HGB) are NOT parallel with Waves 3-7
(RF) because TASK-18/19 append to the SAME `crates/mlrs-py/src/estimators/ensemble.rs`
and `crates/mlrs-py/python/mlrs/ensemble.py` files TASK-08/09/11/12 create —
this mirrors metrics-surface's PLAN-CHECK.md Issue 4 lesson (two tasks
editing the same file must not be marked parallel); TASK-18 is sequenced
strictly after TASK-09 lands (not merely after Wave 3 "starts").

---

## TASK-01 — RF-IMP-01: `rf_node_sqsum` kernel + `rf_best_split` node_decrease output + `RfFitOutcome`

- **Spec:** `RF-IMP-01`
- **Order:** 1 (Wave 1)
- **Depends on:** none
- **Parallel with:** none (foundation)

### Objective
After this task, `rf_fit_impl` (classifier AND regressor target modes)
returns `RfFitOutcome<F> { model: RfModel<F>, feature_importances: Vec<F>,
oob_score: Option<F> }` (with `oob_score` always `None` — RF-OOB-01/TASK-04
fills it in later) with `feature_importances` a normalized (sums to 1),
length-`n_features` vector, computed via a new `rf_node_sqsum` device kernel
+ an extended `rf_best_split` output array + a one-time host reduction.

### Specification References
- `SPEC-RF-IMP-01` — Rust-core `feature_importances()` computation contract.

### Context and Evidence
- `crates/mlrs-kernels/src/tree.rs:262-286` (`rf_node_max`) — the exact
  per-`(tree_in_chunk,node)` reduction shape `rf_node_sqsum` mirrors (running
  accumulator over `nc` classes at the last cumulative bin), `[VERIFIED: LOCAL]`.
- `crates/mlrs-kernels/src/tree.rs:296-347` (`rf_split_scores_class`),
  `:355-387` (`rf_split_scores_reg`), `:403-512` (`rf_best_split`) —
  `[VERIFIED: LOCAL — full read]`, the `best`/`tot`/`hbase` local variables
  this task exposes as `node_decrease`.
- `crates/mlrs-backend/src/prims/random_forest.rs:651-668` (K3/K4 launch
  sites, the insertion point for the new K4.5 `rf_node_sqsum` launch),
  `:670-751` (K5/K6 launch sites, where `rf_best_split`'s `ArrayArg` list
  grows one entry), `:753-757` (transient-buffer release, the pattern the
  new `nsq_h` follows), `:826-836` (`Ok(RfModel{...})`, the return-type
  change point). `[VERIFIED: LOCAL — full read]`
- Derivation of `decrease = score − parent_sumsq/tot` — see "Resolved
  planning decisions" above.

### Files
- Modify: `crates/mlrs-kernels/src/tree.rs` (new `rf_node_sqsum` fn; extend
  `rf_best_split`'s signature with `node_decrease: &mut Array<F>`)
- Modify: `crates/mlrs-backend/src/prims/random_forest.rs` (new K4.5 launch
  + transient buffer in the level loop; new persistent `node_decrease`
  array; new `RfFitOutcome<F>` struct; `rf_fit_impl`'s return type and final
  block change; the host reduction to `feature_importances: Vec<F>`)
- Modify: `crates/mlrs-backend/tests/random_forest_test.rs` — **REQUIRED,
  not optional** (Plan-Check Pass-1 Issue 1). `rf_fit_class`/`rf_fit_reg`'s
  return-type change from `RfModel<F>` to `RfFitOutcome<F>` has exactly 4
  call-site files workspace-wide; this file has **10** direct call sites
  (`[VERIFIED: LOCAL grep rf_fit_class\|rf_fit_reg
  crates/mlrs-backend/tests/random_forest_test.rs` → lines `99, 160, 234,
  238, 258, 260, 262, 265, 269, 274`]`) that bind the result straight as
  `RfModel<F>` (e.g. `let model = rf_fit_class::<F>(...).expect(...);
  let feats = model.split_feature_host(&pool);` at `:99-103`; `let model =
  rf_fit_reg::<F>(...)` at `:160-165`; `rf_predict_proba::<F>(&mut pool,
  &model, ...)`/`rf_predict_reg::<F>(&mut pool, &model, ...)` at `:128,178,235,239,276`
  — both predict fns take `&RfModel<F>`, not `&RfFitOutcome<F>`). Every one
  of these 10 sites must destructure `RfFitOutcome { model, .. }` (or bind
  `.model` field access, e.g. `let outcome = rf_fit_class::<F>(...).expect(...);
  let model = outcome.model;`) BEFORE the existing `RfModel`-typed
  accessor/predict calls compile again. Without this edit, TASK-01's Green
  step leaves `cargo test -p mlrs-backend --features cpu` **failing to
  compile** (not merely failing an assertion) — this file's compile-cleanliness
  is part of TASK-01's own Green step, not a later task's problem.
- Create: `crates/mlrs-backend/tests/random_forest_feature_importances_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `feature_importances_dominant_feature_ranks_highest` in the new
  test file.
- Setup: hand-built 40-row × 2-feature dataset (feature 0 perfectly
  separates 2 classes via a threshold; feature 1 is uniform noise
  uncorrelated with the label), `RfParams { n_trees: 4, max_depth: 3, n_bins:
  8, max_features: 2, bootstrap: false, min_samples_split: 2.0,
  min_samples_leaf: 1.0, seed: 42 }` (Plan-Check Pass-2 fix: `RfParams` at
  THIS task's point in the plan has ONLY its ORIGINAL fields — `n_trees,
  max_depth, n_bins, max_features, min_samples_split, min_samples_leaf,
  bootstrap, seed` — no `oob_score`; TASK-04 (Wave 2, three tasks later) is
  the first task that adds `oob_score` to `RfParams`, both in its own Red
  test literal and, via its "audit every existing construction site" Green
  step, everywhere else `RfParams` is constructed at that later point in
  time. This Red test's FIRST failure is a compile error on the
  not-yet-existing `RfFitOutcome` return type / `feature_importances`
  field, matching the "Expected initial failure" contract below).
- Call `rf_fit_class::<f64>(&mut pool, &x, (40, 2), &y_idx, 2, &params)`
  (existing entry point per `random_forest.rs:150`+) and read
  `outcome.feature_importances`.
- Expected: length `2`, sums to `1.0 ± 1e-9`, `feature_importances[0] >
  feature_importances[1]`.
- Expected initial failure: compile error — `RfFitOutcome`/`.feature_importances`
  do not exist; `rf_fit_class` still returns bare `RfModel<F>`.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_feature_importances_test feature_importances_dominant_feature_ranks_highest`

#### 2. Green
- Add `rf_node_sqsum::<F: Float + CubeElement>` to `tree.rs`, mirroring
  `rf_node_max`'s exact index arithmetic (`base = ((tid)*mf*nb+(nb-1))*nc`,
  loop `c in 0..nc`) but accumulating `sq += v*v` instead of running-max.
- Extend `rf_best_split`'s parameter list with `node_decrease: &mut
  Array<F>` and (classifier arm) a new `node_sq: &Array<F>` input; inside the
  function body, after `is_leaf[midx]` is decided: if `leaf==0`, write
  `node_decrease[midx] = best − node_sq[tid]/tot` (classifier, `mode_class==1`)
  or `best − (hist[hbase+1]*hist[hbase+1])/tot` (regressor, reusing the
  already-computed `hbase`); if `leaf==1`, write `node_decrease[midx] =
  F::new(0.0)`.
- In `rf_fit_impl`'s level loop, add the K4.5 `rf_node_sqsum` launch
  (classifier target only — for `RfTarget::Reg`, pass a zero-filled or
  unused placeholder `nsq_h` since `rf_best_split`'s regressor arm never
  reads `node_sq`; simplest: always allocate `nsq_h` at `stats_len`, launch
  `rf_node_sqsum` unconditionally for BOTH targets reading the same
  `hist_h`/`ncs` layout — `ncs` is `2` for regression, and `rf_node_sqsum`'s
  loop over `0..ncs` on the 2-slot `(n, Σy)` histogram would compute `n² +
  (Σy)²`, NOT what the regressor branch of `rf_best_split` needs — so the
  regressor arm of `rf_best_split` must NOT read `node_sq` at all, matching
  the Context note above; guard this with a `mode_class == 1u32` branch
  inside `rf_best_split` exactly like the existing `if mode_class == 1u32 {
  ... }` pattern at `tree.rs:465-469`). Release `nsq_h` alongside
  `ntot_h`/`nmax_h`.
- Add `node_decrease` as a new persistent `DeviceArray<ActiveRuntime, F>`
  (`t * total_nodes`), allocated alongside `split_feature`/`is_leaf` and
  passed into the K6 launch call.
- Add `pub struct RfFitOutcome<F> { pub model: RfModel<F>, pub
  feature_importances: Vec<F>, pub oob_score: Option<F> }` to
  `random_forest.rs`; change `rf_fit_impl`'s return type from
  `Result<RfModel<F>, PrimError>` to `Result<RfFitOutcome<F>, PrimError>`.
  `RfModel<F>` gains the `node_decrease` field (device-resident, unused by
  any existing accessor — additive, no existing `RfModel` consumer breaks).
- After the level loop, before constructing `RfFitOutcome`: readback
  `split_feature`, `is_leaf`, `node_decrease` (`to_host(pool)` each); host
  loop `for i in 0..t*total_nodes { if is_leaf_host[i]==0 { imp[split_feature_host[i]
  as usize] += host_to_f64(node_decrease_host[i]) } }`; normalize (divide by
  `imp.iter().sum()`, or leave all-zero if the sum is `0.0`); cast back to
  `Vec<F>`.
- Update `rf_fit_class`/`rf_fit_reg` (the two existing public entry points
  that call `rf_fit_impl` — confirm exact names via
  `crates/mlrs-backend/src/prims/random_forest.rs` beyond line 150, Green-time
  grep) to propagate the new return type unchanged (`oob_score: None` always,
  at this task).
- **Fix all 10 call sites in `crates/mlrs-backend/tests/random_forest_test.rs`**
  (lines `99, 160, 234, 238, 258, 260, 262, 265, 269, 274` — the exact list
  cited in this task's Files section) so the file compiles again against the
  new `RfFitOutcome<F>` return type: at each `let model = rf_fit_class::<F>(...)`/
  `let m1 = rf_fit_class::<f32>(...)`/`let model = rf_fit_reg::<F>(...)`
  binding, either destructure `let RfFitOutcome { model, .. } =
  rf_fit_class::<F>(...).expect(...)` or bind the outcome and access
  `.model` at each downstream use (`model.split_feature_host(&pool)`,
  `rf_predict_proba::<F>(&mut pool, &model, ...)`,
  `rf_predict_reg::<F>(&mut pool, &model, ...)` — all three take
  `&RfModel<F>`, unchanged, so only the BINDING at the call site changes,
  not these downstream signatures). The `assert!(rf_fit_class::<f32>(...).is_err())`
  sites (`258, 260, 262, 265, 269`) need NO change (`Result::is_err()` does
  not care which `Ok` payload type it wraps). This is part of THIS task's
  Green step, not deferred to any later task.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_feature_importances_test feature_importances_dominant_feature_ranks_highest`

#### 3. Refactor
- Confirm `node_decrease`'s device buffer is released/dropped correctly on
  every early-return `PrimError` path in `rf_fit_impl` (mirror the existing
  drop discipline for `split_bin`/`edges_dev`/`w_dev` at `:822-824`) — no
  leaked pool allocation on an error path.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_feature_importances_test`

#### 4. Verify
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_feature_importances_test`
- Run: `cargo test -p mlrs-backend --features cpu` (full regression —
  **CORRECTED (Plan-Check Pass-1 Issue 1): `random_forest_test.rs`'s 10
  call sites ARE affected by the return-type change and MUST be updated as
  part of this task's own Green step (see Files/Green above) for this
  command to even compile; its ASSERTED VALUES are unaffected since
  `RfModel`'s existing fields/accessors are untouched — only the BINDING
  shape at each call site changes**)
- Run: `cargo build -p mlrs-backend --features wgpu` (kernel compiles on the
  second backend gate; do not device-execute here, cpu is the primary
  correctness gate per project convention)
- Confirm: `feature_importances` sums to `1.0` within `1e-9` on the Red
  test's synthetic data; regressor mirror test (Implementation Step 2 below)
  passes too.
- Confirm: `crates/mlrs-backend/tests/random_forest_test.rs` compiles AND
  its existing assertions (all pre-dating this task) still pass unchanged —
  a genuine regression check, not merely a compile check.

### Implementation Steps
1. Write the classifier Red test above.
2. Write a second Red test, `feature_importances_regressor_dominant_feature`
   — same 2-feature synthetic geometry but continuous `y` strongly
   correlated with feature 0 only, calling `rf_fit_reg`.
3. Write a third Red test, `feature_importances_all_leaf_forest_is_all_zero`
   — `max_depth: 0`-equivalent-degenerate forced-leaf construction (or the
   smallest `max_depth` the builder allows, `1`, combined with `min_samples_split`
   set high enough that every node is forced to a leaf) asserting
   `feature_importances.iter().all(|v| *v == 0.0)` and no panic (the
   divide-by-zero guard).
4. Implement `rf_node_sqsum` + `rf_best_split` extension + `rf_fit_impl`
   changes to pass all three at once.
5. Fix all 10 `RfModel`-binding call sites in
   `crates/mlrs-backend/tests/random_forest_test.rs` (lines `99, 160, 234,
   238, 258, 260, 262, 265, 269, 274`) to destructure/access `.model` from
   the new `RfFitOutcome<F>` return type (Plan-Check Pass-1 Issue 1 — see
   Files/Green above for the exact per-site pattern).
6. Run the full `mlrs-backend` regression suite, confirming
   `random_forest_test.rs` both COMPILES and its pre-existing assertions
   still pass unchanged.

### Completion Criteria
- [ ] All 3 Red tests fail for the stated reason (compile error /
      not-yet-existing field) before Green.
- [ ] All 3 pass after Green.
- [ ] `cargo test -p mlrs-backend --features cpu` full suite COMPILES and
      passes — INCLUDING `random_forest_test.rs`'s 10 updated call sites
      (Plan-Check Pass-1 Issue 1; this file's update is part of THIS task,
      not deferred).
- [ ] `node_decrease` device buffer has no leaked-pool path.
- [ ] `RfFitOutcome.oob_score` is always `None` at the end of this task
      (RF-OOB-01/TASK-04 is the only place that ever sets `Some`).

### Risks and Guardrails
- Risk: `rf_node_sqsum`'s classifier-only semantics being accidentally
  launched/read for the regressor target, producing a nonsensical `n² +
  (Σy)²` value. Mitigation: `rf_best_split`'s regressor arm must gate on
  `mode_class == 1u32` before ever indexing `node_sq`, exactly like the
  existing purity check at `tree.rs:465-469` — the Red test #2 (regressor)
  is the guardrail that would catch a leaked classifier-only value.
- Risk: forgetting to release the new `nsq_h`/`node_decrease` buffers on an
  error path, silently violating the FOUND-05/D-10 memory-conservation
  invariant. Mitigation: the Refactor step's explicit early-return audit.
- Risk (Plan-Check Pass-1 Issue 1, CRITICAL, now resolved): the return-type
  change silently breaking `crates/mlrs-backend/tests/random_forest_test.rs`'s
  10 pre-existing call sites, leaving `cargo test -p mlrs-backend --features
  cpu` unable to COMPILE after this task's Green step. Mitigation: this
  file is now an explicit `Modify` target in Files/Green/Implementation
  Steps/Completion Criteria above — not a task boundary this task can defer.

---

## TASK-02 — RF-IMP-01: `RandomForestClassifier::feature_importances()` + oracle fixture extension

- **Spec:** `RF-IMP-01`
- **Order:** 2 (Wave 1)
- **Depends on:** TASK-01
- **Parallel with:** none (owns the shared `gen_oracle.py` edit; TASK-03
  consumes it)

### Objective
`RandomForestClassifier<F, Fitted>::feature_importances() -> &[F]` exists,
populated at `fit()` time from `RfFitOutcome`, matching sklearn's
`feature_importances_` within `atol=0.05` on the existing deterministic-tier
fixture (now carrying a new `ref_feature_importances` key), plus a
qualitative dominant-feature-ranking assertion as the PRIMARY correctness
signal.

**[RESOLVED at Green time, 2026-07-18 — SPEC.md §5/§9 updated to match]:**
the original `1e-5` exact-match claim below is superseded. `predict`/
`predict_proba` exact-match (the existing deterministic-tier precedent) only
proves outcome-equivalence, not split-choice-equivalence: sklearn's own
splitter breaks near-tied candidate splits using internal state independent
of the public seed, so sklearn's own two deterministic-tier trees are NOT
bit-identical to each other even though mlrs's are — confirmed empirically
(`det.estimators_[0].feature_importances_ != det.estimators_[1]...`), with a
genuine tied-split divergence at a low-sample deep node producing a ~0.0022
per-feature `feature_importances_` gap. `atol=0.05` (25x the observed gap)
replaces `1e-5`/`1e-4` throughout this task's Red/Green/Verify steps below.

### Specification References
- `SPEC-RF-IMP-01` — algos-layer accessor + oracle-parity contract.

### Context and Evidence
- `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:132-160`
  (`impl<F> RandomForestClassifier<F, Fitted>`, the block this task adds
  `feature_importances_: Vec<F>` + `feature_importances()` to) and
  `:366-419` (the `Fit::fit` method — not directly read this pass beyond its
  line range citation in SPEC.md, `[VERIFIED: LOCAL SPEC.md §3, CODEGRAPH]` —
  Green-time must confirm it currently calls `rf_fit_class(...)` and
  destructure the new `RfFitOutcome` there).
- `crates/mlrs-algos/tests/random_forest_classifier_test.rs:36-192` — the
  existing deterministic/statistical two-tier fixture consumer this task's
  new oracle test sits alongside (`[VERIFIED: LOCAL]`).
- `scripts/gen_oracle.py` — the existing `gen_rf_classifier` generator
  (exact name/line TBD, confirm via `grep -n "def gen_rf" scripts/gen_oracle.py`
  at Green time) — extended, not replaced.

### Files
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
  (new `feature_importances_` field on `Fitted`, new accessor, `fit()`
  destructures `RfFitOutcome`)
- Modify: `crates/mlrs-algos/tests/random_forest_classifier_test.rs` (new
  test functions)
- Modify: `scripts/gen_oracle.py` (extend the existing RF classifier
  generator with `ref_feature_importances`, computed via
  `sklearn.ensemble.RandomForestClassifier(...).feature_importances_` on the
  SAME deterministic-tier construction args the existing generator already
  uses)
- Modify (regenerated data, not source):
  `tests/fixtures/rf_cls_f32_seed42.npz`, `tests/fixtures/rf_cls_f64_seed42.npz`
  (new key added, existing keys unchanged)

### TDD Sequence

#### 1. Red
- Test name: `feature_importances_matches_sklearn_deterministic_f64` in
  `random_forest_classifier_test.rs`.
- Setup: load `rf_cls_f64_seed42.npz` (existing fixture, once regenerated
  with the new key), build the deterministic-tier classifier
  (`bootstrap(false).max_features(MaxFeatures::All)...` — mirror the exact
  existing deterministic-tier construction at `random_forest_classifier_test.rs:82-120`),
  `.fit(...)`.
- Call `.feature_importances()`.
- Expected: `(got[i] - ref_feature_importances[i]).abs() < 0.05` for every
  feature `i` (deterministic tier — `atol=0.05`, NOT `1e-5`; see the
  "RESOLVED at Green time" note in this task's Objective — `predict`/
  `predict_proba` structural identity does not extend to split-tie-breaking,
  which sklearn's own splitter resolves with seed-independent internal
  state).
- Expected initial failure: compile error — `feature_importances()` method
  does not exist on `RandomForestClassifier<F, Fitted>`.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test feature_importances_matches_sklearn_deterministic_f64`

#### 2. Green
- Add `feature_importances_: Vec<F>` to the `Fitted`-state struct fields (or
  the `RandomForestClassifier<F, Fitted>`-only field block, following the
  existing `model_`/`classes_` field pattern at
  `random_forest_classifier.rs:56-78`).
- In `fit()`, change the call from `rf_fit_class(...)` (returning `RfModel`)
  to destructure the new `RfFitOutcome { model, feature_importances,
  oob_score: _ }` (this task ignores `oob_score`, TASK-06 wires it).
- Add `pub fn feature_importances(&self) -> &[F] { &self.feature_importances_
  }` to the `impl<F> RandomForestClassifier<F, Fitted>` block.
- In `gen_oracle.py`, add one `ref_feature_importances` array to the
  existing deterministic-tier RF classifier generation (via
  `sklearn.ensemble.RandomForestClassifier` constructed with the SAME
  deterministic-tier kwargs the file already uses for `det_pred_train` —
  confirm the exact kwarg block at Green time), cast to the fixture's own
  dtype, appended to the SAME `np.savez(...)` call the existing generator
  already makes (additive key, not a new file).
- Regenerate `rf_cls_f32_seed42.npz`/`rf_cls_f64_seed42.npz` in the
  established oracle venv; commit (staged) alongside this task's other
  changes.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test feature_importances_matches_sklearn_deterministic_f64`

#### 3. Refactor
- Confirm the existing `det_pred_train`/`det_proba_train`/`stat_acc_test`
  assertions in `random_forest_classifier_test.rs` still pass unchanged
  (regenerating the fixture must be byte-identical on every PRE-EXISTING
  key — `np.savez` with the same inputs plus one new array does not perturb
  existing arrays).
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`
- Run: `cargo test -p mlrs-algos --features cpu` (full regression)
- Confirm: `feature_importances().iter().sum::<f64>()` (cast) is `1.0 ± 1e-9`
  on the fixture-driven fit, independently of the `atol=0.05` sklearn-comparison
  assertion (a second, cheaper sanity check).

### Implementation Steps
1. Write the Red test (f64, deterministic tier, `atol=0.05`).
2. Write a companion f32 Red test at `atol=0.05` (same tolerance as f64 for
   this specific assertion — the dominant error source is sklearn's
   tie-break nondeterminism, ~0.0022, not f32-vs-f64 rounding, so no
   separate wider f32 band is needed here; this deliberately departs from
   the "established f32 tolerance convention" of a tighter dtype-only gap
   like `1e-4`, because the gap this test tolerates is not a dtype-rounding
   gap).
3. Write a qualitative Red test, `feature_importances_dominant_feature_ranking`,
   using a NEW small hand-built dataset (not a fixture — mirrors TASK-01's
   Rust-level synthetic data but exercised through the PUBLIC algos API) on
   the STATISTICAL-tier hyperparameters (`bootstrap=true`, defaults),
   asserting only the RANKING property (`importances[dominant] >
   importances[noise_i]` for every noise feature), never an exact match —
   resolves SPEC's "acceptance-tolerance strategy" unresolved question.
4. Extend `gen_oracle.py`'s existing RF classifier generator; regenerate;
   confirm via `git diff --stat tests/fixtures/rf_cls_*.npz` that only the
   new key changed (not a full-file rewrite that would also perturb
   existing bytes — if the regen changes existing keys' bytes, STOP: it
   means the generator's random seed/consumption order shifted, a bug, not
   an acceptable diff).
5. Implement the accessor + `fit()` change.
6. Run the full regression suite.

### Completion Criteria
- [ ] All 3+ Red tests fail for the stated reason before Green.
- [ ] All pass after Green.
- [ ] Every PRE-EXISTING fixture assertion in `random_forest_classifier_test.rs`
      still passes unchanged (regen did not perturb existing keys).
- [ ] `feature_importances()` sums to `1.0` on every constructed forest in
      this task's tests (deterministic, statistical, and the all-leaf
      degenerate case from TASK-01).

### Risks and Guardrails
- Risk: regenerating the fixture accidentally changes `det_pred_train`'s
  bytes (e.g. if the generator's RNG stream shifted). Mitigation: the
  Implementation Step 4 `git diff --stat` / per-key byte-diff check.
- **[RESOLVED at Green time, 2026-07-18]** Risk (as originally written):
  sklearn's `feature_importances_` computes a per-TREE-normalized mean (not
  the "sum-then-normalize-once" SPEC explicitly picked) — these CAN diverge
  numerically on trees with different total node-decrease scale. **Outcome:**
  this was NOT the actual divergence source — on the deterministic tier
  mlrs's trees ARE bit-identical, which makes sum-then-normalize-once and
  per-tree-normalize-then-average PROVABLY equal (verified algebraically:
  if every tree's total raw-decrease is identical, the two formulas reduce
  to the same value). The REAL divergence source is sklearn's own
  splitter breaking near-tied splits with seed-independent internal state,
  so sklearn's own 2 deterministic-tier trees are not bit-identical to each
  other. Per the "STOP and re-derive" instruction below, execution stopped
  and reported this as a specification conflict rather than silently
  patching the tolerance; the orchestrator resolved it by replacing the
  `1e-5` exact-match tier with `atol=0.05` (SPEC.md §5/§9, spec_revision 2)
  — see this task's Objective for the full resolution. The qualitative
  ranking test (Implementation Step 3) is unaffected and remains the
  PRIMARY signal.

---

## TASK-03 — RF-IMP-01: `RandomForestRegressor::feature_importances()`

- **Spec:** `RF-IMP-01`
- **Order:** 3 (Wave 1)
- **Depends on:** TASK-02 (consumes the `gen_oracle.py`/fixture edit TASK-02
  already made — no second edit to the same generator function)
- **Parallel with:** none

### Objective
`RandomForestRegressor<F, Fitted>::feature_importances() -> &[F]` mirrors
TASK-02 for the regressor.

**[RESOLVED at TASK-02 Green time, 2026-07-18 — applies here too]:** use
`atol=0.05` (NOT `1e-5`/`1e-4`) for the deterministic-tier sklearn-comparison
assertion, both f32 and f64 — see TASK-02's Objective/Risks for the full
root-cause (sklearn's own splitter tie-break nondeterminism, not an mlrs
formula bug). The qualitative ranking test remains the primary signal.

### Specification References
- `SPEC-RF-IMP-01` — regressor half of the same contract.

### Context and Evidence
- `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` — mirrors
  `random_forest_classifier.rs`'s `Fitted`-state field/accessor block
  structurally (`[VERIFIED: CODEGRAPH — cited in SPEC.md §3, defaults at
  RESEARCH.md §5.1]`); exact field layout confirmed at Green time by reading
  the file (not yet read by this Planner beyond SPEC's citation — flagged
  here rather than assumed).
- `tests/fixtures/rf_reg_{f32,f64}_seed42.npz` — extended by TASK-02's
  `gen_oracle.py` edit (the SAME task added the regressor's
  `ref_feature_importances` key alongside the classifier's, per TASK-02
  Implementation Step 4's scope — confirm this at Green time; if TASK-02
  only touched the classifier generator, this task's Green step extends the
  regressor generator itself, still a single `gen_oracle.py` edit owned
  here, sequenced after TASK-02 to avoid the same-file collision).

### Files
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs`
- Modify: `crates/mlrs-algos/tests/random_forest_regressor_test.rs`
- Modify (if not already done by TASK-02): `scripts/gen_oracle.py`,
  `tests/fixtures/rf_reg_{f32,f64}_seed42.npz`

### TDD Sequence

#### 1. Red
- Test name: `feature_importances_matches_sklearn_deterministic_f64` in
  `random_forest_regressor_test.rs`, mirroring TASK-02's Red test exactly
  (regressor fixture, `sklearn.ensemble.RandomForestRegressor.feature_importances_`
  reference).
- Expected initial failure: compile error — method does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test feature_importances_matches_sklearn_deterministic_f64`

#### 2. Green
- Mirror TASK-02's accessor + `fit()` change for the regressor struct.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test feature_importances_matches_sklearn_deterministic_f64`

#### 3. Refactor
- Confirm no duplicated logic between the classifier/regressor
  `feature_importances()` bodies beyond what the type system already forces
  (both are one-line field accessors — no further extraction needed).
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test`
- Run: `cargo test -p mlrs-algos --features cpu` (full regression)

### Implementation Steps
1. Write the f64 deterministic Red test + f32 companion + the qualitative
   ranking test (mirrors TASK-02 Implementation Steps 2-3, regressor
   variant: continuous `y` strongly correlated with one feature).
2. Confirm/extend `gen_oracle.py`'s regressor generator (only if TASK-02
   did not already cover it).
3. Implement the accessor.
4. Run the full regression suite.

### Completion Criteria
- [ ] All Red tests fail for the stated reason before Green.
- [ ] All pass after Green.
- [ ] No existing regressor fixture assertion regresses.

### Risks and Guardrails
- **[RESOLVED, see TASK-02]** Same numeric-tolerance risk as TASK-02
  (originally framed as per-tree vs. sum-once normalization; actual root
  cause was sklearn's own splitter tie-break nondeterminism) — same
  resolution: `atol=0.05` deterministic-tier assertion, qualitative ranking
  test as the primary signal, not an exact match.

---

## TASK-04 — RF-OOB-01: bootstrap-rederive + OOB aggregation in `rf_fit_impl`

- **Spec:** `RF-OOB-01`
- **Order:** 4 (Wave 2)
- **Depends on:** TASK-01 (same function, same file — `RfFitOutcome` must
  already exist)
- **Parallel with:** none

### Objective
`RfParams` gains `pub oob_score: bool`; when `true`, `rf_fit_impl` computes
`RfFitOutcome.oob_score = Some(score)` (accuracy for classifier, R² for
regressor) using ONLY out-of-bag trees per training row, via a rederived
bootstrap mask + the existing `rf_predict_leaf` kernel; when `false` (the
default), zero extra device/host work happens beyond today.

### Specification References
- `SPEC-RF-OOB-01` — Rust-core OOB computation contract.

### Context and Evidence
- `crates/mlrs-backend/src/prims/random_forest.rs:379-397`
  (`bootstrap_weights`), `:466-470` (the fixed RNG consumption order),
  `:826-836` (the `Ok(RfModel{...})`/return-type point this task extends
  further) — `[VERIFIED: LOCAL — full read]`.
- `crates/mlrs-kernels/src/tree.rs:636-670` (`rf_predict_leaf`, the reused
  traversal kernel), `:677-701` (`rf_vote_class`, NOT reused directly — this
  task's host-side OOB averaging is a masked variant of the same
  mean-of-leaf-distributions idea, done host-side since only a SUBSET of
  trees per row participates, which the device kernel's unconditional
  `t in 0..n_trees` loop does not support without a new kernel; doing this
  reduction host-side avoids a THIRD new kernel for what is already a
  one-time, `O(t·n)`-scale, non-hot-loop computation). `[VERIFIED: LOCAL]`
- "Resolved planning decisions" above — the full mechanism.

### Files
- Modify: `crates/mlrs-backend/src/prims/random_forest.rs` (`RfParams.oob_score`
  field; the gated OOB block in `rf_fit_impl`)
- Create: `crates/mlrs-backend/tests/random_forest_oob_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `oob_score_false_is_none_and_adds_no_cost` in the new test
  file.
- Setup: any valid small dataset, `RfParams { oob_score: false, ... }`
  (compile error initially — the field does not exist).
- Expected: `outcome.oob_score.is_none()`.
- Expected initial failure: compile error — `RfParams` has no `oob_score`
  field.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_oob_test oob_score_false_is_none_and_adds_no_cost`

#### 2. Green
- Add `pub oob_score: bool` to `RfParams`.
- Gate the entire OOB block behind `if params.oob_score { ... } else { None
  }`, assigned to `RfFitOutcome.oob_score`.
- Inside the `true` branch: rederive `w_host` via a fresh
  `SplitMix64::new(params.seed)` + `bootstrap_weights::<F>`; launch
  `rf_predict_leaf` on `x` against the just-built `split_feature`/`threshold`/`is_leaf`
  (still local variables, not yet moved into `RfModel`); readback
  `leaf_ids`, `leaf_dist`, and the target `y_dev` (`RfTarget::Class(y_dev,_)`
  → dense class indices; `RfTarget::Reg(y_dev)` → continuous targets); host
  loop per row `i`: collect `leaf_dist` rows for trees `t` where
  `w_host[t*n+i]==0`; if the OOB tree count for row `i` is `0`, skip the row
  and increment a `zero_oob_count`; else average (mean-of-leaf-distributions
  for classifier → argmax vs. dense class index for accuracy; mean leaf
  value for regressor → accumulate for R²); after the loop, if
  `zero_oob_count > 0`, `log::warn!(...)`; compute the final accuracy/R²
  over the NON-skipped rows only.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_oob_test oob_score_false_is_none_and_adds_no_cost`

#### 3. Refactor
- Factor the classifier-vs-regressor OOB-averaging branch to share the
  "collect OOB tree indices per row" loop (identical for both modes; only
  the final score formula differs) — small, local refactor, no behavior
  change.
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_oob_test`

#### 4. Verify
- Run: `cargo test -p mlrs-backend --features cpu --test random_forest_oob_test`
- Run: `cargo test -p mlrs-backend --features cpu` (full regression — `oob_score:
  false` is the default in every OTHER existing test's `RfParams`
  construction; confirm every existing test-site literal still compiles,
  i.e. this task must NOT require every existing `RfParams { ... }`
  construction site to be updated if they use `..Default::default()` —
  confirm `RfParams` has/needs a `Default` impl at Green time, or that
  every existing construction site is a full explicit-field literal that
  now needs one more field; if the latter, enumerate and fix every site as
  part of Green, not left broken).

### Implementation Steps
1. Write the `false` Red test.
2. Write a second Red test, `oob_score_true_matches_statistical_band` —
   loads the (not-yet-extended) `rf_cls_f64_seed42.npz` statistical-tier
   construction args, `oob_score: true, bootstrap: true`; asserts
   `outcome.oob_score.is_some()` and a WIDE placeholder band (e.g. `0.0..=1.0`)
   at THIS task's Green (the exact `ref_oob_score` cross-check against
   sklearn is TASK-06/07's job, not this task's — this task proves the
   MECHANISM produces a plausible in-range score, not sklearn parity yet).
3. Write a third Red test, `oob_score_zero_oob_rows_excluded_not_panicking`
   — a pathologically tiny forest (`n_trees: 1`) where some row is
   guaranteed never out-of-bag; assert no panic and a finite score.
4. Implement the `RfParams` field + gated OOB block.
5. Audit every existing `RfParams { ... }` construction site across
   `mlrs-backend`/`mlrs-algos` for the new required field; add `oob_score:
   false` to each (or add `impl Default for RfParams` if that is the
   established convention for this struct — confirm at Green time; SPEC
   does not currently show `RfParams` deriving `Default`, so an explicit
   per-site addition is the conservative default action).
6. Run the full regression suite.

### Completion Criteria
- [ ] All 3 Red tests fail for the stated reason before Green.
- [ ] All 3 pass after Green.
- [ ] `cargo test -p mlrs-backend --features cpu` full suite green (no
      broken existing `RfParams` literal).
- [ ] Zero-OOB-row case does not panic; `log::warn!` fires exactly once per
      `fit()` call when triggered (not once per row).

### Risks and Guardrails
- Risk: forgetting to update EVERY existing `RfParams` construction site
  when the struct gains a field (a plain-Rust compile-time guardrail — the
  build itself fails if any site is missed, so this risk is self-detecting,
  not silent).
- Risk: the OOB host loop reading `leaf_dist`/`leaf_ids` INDEXED by the
  WRONG node id if `rf_predict_leaf` is launched against arrays that have
  already been partially reused/released — mitigation: launch it BEFORE any
  `release_into(pool)` call on `split_feature`/`threshold`/`is_leaf`/`leaf_dist`
  (i.e., insert the OOB block between the level loop's end (`:812`) and the
  "Fit-only scratch back to the pool" release block (`:816-824`), NOT after).

---

## TASK-05 — RF-OOB-01: `oob_score=true, bootstrap=false` builder rejection

- **Spec:** `RF-OOB-01`
- **Order:** 5 (Wave 2)
- **Depends on:** TASK-04
- **Parallel with:** none

### Objective
`RandomForestClassifierBuilder::oob_score(bool)` /
`RandomForestRegressorBuilder::oob_score(bool)` setters exist (default
`false`); `build::<F>()` returns `Err(BuildError::OobRequiresBootstrap {
estimator })` when `oob_score=true, bootstrap=false`, matching sklearn's
`ValueError`. **Explicitly (Plan-Check Pass-1 Issue 4): the main
`RandomForestClassifier<F, S>`/`RandomForestRegressor<F, S>` STRUCT ITSELF
gains its own `oob_score: bool` field — not merely the Builder — mirroring
the existing `bootstrap: bool` field exactly (`random_forest_classifier.rs:56-78`,
which the struct already carries `bootstrap` on, separate from
`RandomForestClassifierBuilder`'s own same-named field). `build::<F>()`
must populate this field on the returned `RandomForestClassifier<F, Unfit>`
(`self.oob_score` from the builder → `oob_score: self.oob_score` in the
`Ok(RandomForestClassifier { ... })` literal at
`random_forest_classifier.rs:250-264`, exactly parallel to how
`bootstrap: self.bootstrap` is already threaded there today) — WITHOUT this,
TASK-06's `fit()` cannot read `self.oob_score` to thread it into
`RfParams.oob_score`, since a value that lives only on the (consumed,
dropped-at-`build()`) Builder is not available inside `fit()`, which is
called on the already-built `Unfit`/`Fitted`-transitioning struct, not on
the Builder.

### Specification References
- `SPEC-RF-OOB-01` — build-time validation contract.

### Context and Evidence
- `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:182-265`
  (`RandomForestClassifierBuilder`, the exact setter/`build::<F>()` block
  this task extends — `[VERIFIED: LOCAL — full read]`), `:267+`
  (`validate_forest_hyperparams`, `pub(crate) fn`, confirmed NOT in
  `ensemble/mod.rs` — see "Resolved planning decisions").
- `crates/mlrs-algos/src/error.rs:400+` (`BuildError` enum,
  `InvalidAlpha`/`InvalidL1Ratio`/`InvalidEps` — the exact
  `#[error("...")] Variant { estimator: &'static str, ... }` shape this
  task's new `OobRequiresBootstrap` variant mirrors). `[VERIFIED: LOCAL]`

### Files
- Modify: `crates/mlrs-algos/src/error.rs` (new `BuildError::OobRequiresBootstrap`
  variant)
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
  (`oob_score` field on BOTH `RandomForestClassifierBuilder` AND the main
  `RandomForestClassifier<F, S>` struct itself + setter + `build::<F>()`
  cross-check + `build::<F>()`'s `Ok(...)` literal threading `self.oob_score`
  onto the returned struct's own field — Plan-Check Pass-1 Issue 4)
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` (same,
  mirrored — struct field AND builder field, both)
- Modify: `crates/mlrs-algos/tests/random_forest_classifier_test.rs`,
  `random_forest_regressor_test.rs` (new builder-rejection unit tests)

### TDD Sequence

#### 1. Red
- Test name: `builder_rejects_oob_score_without_bootstrap` in
  `random_forest_classifier_test.rs`.
- Setup: `RandomForestClassifier::<f64>::builder().oob_score(true).bootstrap(false).build::<f64>()`.
- Expected: `matches!(result, Err(BuildError::OobRequiresBootstrap { .. }))`.
- Expected initial failure: compile error — `.oob_score(...)` setter does
  not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test builder_rejects_oob_score_without_bootstrap`

#### 2. Green
- Add `BuildError::OobRequiresBootstrap { estimator: &'static str }` with an
  `#[error("estimator '{estimator}': oob_score=true requires bootstrap=true")]`
  attribute, mirroring `InvalidEps`'s shape exactly.
- Add `oob_score: bool` to **both builder structs** (default `false`,
  matching `RandomForestClassifierBuilder::default()`'s existing
  `RandomForestClassifier::<f64, Unfit>::new().into_builder()` derivation —
  add `oob_score: false` to `new()`'s field list and `into_builder()`'s
  round-trip, per the D-08 single-source discipline this file already
  follows).
- **Add `oob_score: bool` to the MAIN `RandomForestClassifier<F, S>`/
  `RandomForestRegressor<F, S>` struct field lists too (Plan-Check Pass-1
  Issue 4 — a SEPARATE, ADDITIONAL edit from the Builder field above,
  mirroring the existing `bootstrap: bool` struct field at
  `random_forest_classifier.rs:56-78`)**: add it to `new()`'s literal
  (`oob_score: false`), to `into_builder()`'s round-trip
  (`RandomForestClassifierBuilder { ..., oob_score: self.oob_score }`), and
  to `build::<F>()`'s `Ok(RandomForestClassifier { ..., oob_score:
  self.oob_score, model_: None, ... })` literal — every place `bootstrap`
  already appears in these three functions, `oob_score` now appears
  alongside it. This is what makes `self.oob_score` readable inside `fit()`
  (TASK-06), which operates on the struct instance, not the (already
  consumed) Builder.
- Add `pub fn oob_score(mut self, v: bool) -> Self { self.oob_score = v;
  self }` to both builders.
- In each `build::<F>()`, after the existing `validate_forest_hyperparams(...)?`
  call: `if self.oob_score && !self.bootstrap { return
  Err(BuildError::OobRequiresBootstrap { estimator:
  "random_forest_classifier" }); }` (regressor: `"random_forest_regressor"`).
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test builder_rejects_oob_score_without_bootstrap`

#### 3. Refactor
- None needed — the check is a two-line inline addition, no duplication to
  extract beyond what already exists.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test`
- Confirm: `oob_score(true).bootstrap(true).build()` still `Ok(...)` (the
  positive case is not accidentally rejected).

### Implementation Steps
1. Write the classifier Red test.
2. Write the regressor mirror Red test.
3. Write a positive-case regression test,
   `builder_accepts_oob_score_with_bootstrap`, for both estimators.
4. Implement the `BuildError` variant + both builders' setter/cross-check.
5. Run the full regression suite.

### Completion Criteria
- [ ] All Red tests fail for the stated reason before Green.
- [ ] All pass after Green.
- [ ] The positive case (`oob_score=true, bootstrap=true`) is unaffected.
- [ ] The MAIN `RandomForestClassifier<F, S>`/`RandomForestRegressor<F, S>`
      struct (not only its Builder) carries its own `oob_score: bool` field,
      populated by `build::<F>()`, readable from `self.oob_score` inside
      `fit()` (Plan-Check Pass-1 Issue 4) — verified by TASK-06's `fit()`
      change compiling against it without needing to re-thread the value
      through any other path.

### Risks and Guardrails
- Risk: `RandomForestClassifierBuilder::default()`'s
  `into_builder()`/`new()` round-trip missing the new field, silently
  defaulting `oob_score` inconsistently between the two paths. Mitigation:
  Green step explicitly touches BOTH `new()` and `into_builder()`, matching
  every other field in the same struct.
- Risk (Plan-Check Pass-1 Issue 4, now resolved): adding `oob_score` ONLY to
  the Builder (mirroring only half of the existing `bootstrap` pattern)
  would silently leave TASK-06 unable to read the flag inside `fit()`.
  Mitigation: Files/Green above now explicitly call out the MAIN struct
  field as a separate, required edit, and Completion Criteria checks for it.

---

## TASK-06 — RF-OOB-01: `oob_score_` sklearn-parity oracle test (classifier)

- **Spec:** `RF-OOB-01`
- **Order:** 6 (Wave 2)
- **Depends on:** TASK-05
- **Parallel with:** none (owns the `gen_oracle.py` OOB fixture edit;
  TASK-07 consumes it)

### Objective
`RandomForestClassifier::oob_score()` (added in this task, delegating to
`RfFitOutcome.oob_score`) matches sklearn's `oob_score_` within a
STATISTICAL tolerance band on the existing statistical-tier fixture
(extended with `ref_oob_score` + the `oob_score=True` sklearn construction).

### Specification References
- `SPEC-RF-OOB-01` — statistical-tier oracle assertion.

### Context and Evidence
- `crates/mlrs-algos/tests/random_forest_classifier_test.rs` — the existing
  `ACC_MARGIN=0.05` statistical-tier precedent (`[VERIFIED: LOCAL SPEC.md §3
  citation, RESEARCH.md §5.4]`) this task's OOB band reuses/extends.
- SPEC §5 RF-OOB-01 — "statistical-tier-only assertion... no exact-match
  tier is possible for a stochastic quantity" (`SplitMix64` ≠ sklearn
  `MT19937`).

### Files
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
  (`oob_score_: Option<F>` field + `oob_score()` accessor; `fit()` populates
  it from `RfFitOutcome`; builder's `.oob_score(v)` threading into
  `RfParams.oob_score` at the `Fit::fit` call site)
- Modify: `crates/mlrs-algos/tests/random_forest_classifier_test.rs`
- Modify: `scripts/gen_oracle.py` (extend the SAME statistical-tier
  classifier generation this file already performs, adding
  `oob_score=True` to the sklearn construction + `ref_oob_score`, for BOTH
  the `f32` and `f64` dtype passes this generator already makes — Plan-Check
  Pass-1 Issue 2)
- Modify (regenerated data): `tests/fixtures/rf_cls_{f32,f64}_seed42.npz`
  (both dtype files gain `ref_oob_score` — this task's own Files section
  already names BOTH, so the f32 key was always being generated; TASK-06's
  ORIGINAL TDD sequence simply never asserted against it — fixed below)

### TDD Sequence

#### 1. Red
- Test name: `oob_score_within_statistical_band_f64`.
- Setup: load `rf_cls_f64_seed42.npz`'s statistical-tier construction args
  (defaults, `n_estimators=64, depth=8`) plus `.oob_score(true)`, `.fit(X,y)`.
- Expected: `(got - ref_oob_score).abs() < OOB_MARGIN` — a documented band
  (start with `0.10`, WIDER than `ACC_MARGIN=0.05` since OOB compounds two
  sources of stochastic divergence — the bootstrap draw AND which rows are
  OOB per tree — between `SplitMix64` and `MT19937`; this Planner does not
  claim `0.10` is precisely correct, it is a documented starting point to be
  tightened or loosened at Green time against the ACTUAL observed
  divergence, never silently widened past what the Green-time run shows is
  needed).
- Expected initial failure: compile error — `.oob_score()` method /
  `.oob_score(bool)` builder-instance-threading do not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test oob_score_within_statistical_band_f64`
- **Companion Red test (Plan-Check Pass-1 Issue 2 — f32, mandatory, matches
  the TASK-02/03-established f32-and-f64-both-validated convention):** Test
  name `oob_score_within_statistical_band_f32`, identical setup against
  `rf_cls_f32_seed42.npz`, asserting `(got - ref_oob_score).abs() <
  OOB_MARGIN_F32` — a SEPARATE constant from `OOB_MARGIN` (f32's lower
  mantissa precision compounds with the `SplitMix64`/`MT19937` divergence
  differently than f64's; do not assume the same band applies). Same
  "compile error" expected initial failure (both Red tests fail together,
  one Green pass fixes both, exactly like TASK-02/03's f32/f64 pairing).
  Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test oob_score_within_statistical_band_f32`

#### 2. Green
- Add `oob_score_: Option<F>` to the `Fitted`-state fields; `fit()`
  threads `self.oob_score` (the builder-carried flag, stored on the
  `Unfit`→built struct) into `RfParams.oob_score` at the existing
  `rf_fit_class(...)` call site, and destructures
  `RfFitOutcome.oob_score` into `oob_score_`.
- Add `pub fn oob_score(&self) -> Option<F> { self.oob_score_ }`.
- Extend `gen_oracle.py`'s existing statistical-tier RF classifier
  generation with a second sklearn construction (`oob_score=True,
  bootstrap=True`, same seed/data) → `ref_oob_score`, computed ONCE (not
  dtype-dependent — sklearn's own float precision is fixed) and cast into
  BOTH the `f32` and `f64` `np.savez` calls this generator already makes;
  append to the SAME `np.savez` call in each dtype pass.
- Regenerate; run BOTH Red tests; if the observed `|got - ref|` exceeds the
  starting `0.10` band for EITHER dtype, widen that dtype's own documented
  constant independently (never silently, never coupling the two bands
  together) and record the observed value in a code comment.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test oob_score_within_statistical_band_f64`
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test oob_score_within_statistical_band_f32`

#### 3. Refactor
- None beyond confirming the `OOB_MARGIN`/`OOB_MARGIN_F32` constants are
  named and documented (not bare magic numbers) alongside the existing
  `ACC_MARGIN` constant in the same test file.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`
- Run: `cargo test -p mlrs-algos --features cpu` (full regression)
- Confirm: `oob_score(false)` (default) still returns `None` end-to-end
  through the full `builder → fit → accessor` path (not just at the prim
  layer, TASK-04's own test), for BOTH dtypes.
- Confirm: `ref_oob_score` in BOTH `rf_cls_f32_seed42.npz` and
  `rf_cls_f64_seed42.npz` is actually asserted against by a passing test —
  no orphaned fixture key (Plan-Check Pass-1 Issue 2).

### Implementation Steps
1. Write the f64 Red test.
2. Write the f32 companion Red test (Plan-Check Pass-1 Issue 2).
3. Write a companion `oob_score_none_when_flag_false` regression test
   (public-API level, not just TASK-04's prim-level version) — run once, no
   dtype split needed (the `None` case has no numeric tolerance to tune).
4. Implement the accessor + `fit()` threading.
5. Extend `gen_oracle.py`; regenerate both dtype fixtures; tune
   `OOB_MARGIN`/`OOB_MARGIN_F32` independently against their OWN observed
   values.
6. Run the full regression suite.

### Completion Criteria
- [ ] Red tests (BOTH f64 and f32) fail for the stated reason before Green.
- [ ] Both pass after Green, with `OOB_MARGIN` (f64) and `OOB_MARGIN_F32`
      set from OBSERVED Green-time values, not guessed, and not assumed
      identical to each other.
- [ ] `oob_score(false)` (default) end-to-end returns `None`.
- [ ] `ref_oob_score` in the f32 fixture is asserted against (not orphaned —
      Plan-Check Pass-1 Issue 2).

### Risks and Guardrails
- Risk: the `SplitMix64` vs. `MT19937` divergence is large enough that NO
  reasonable band gives a meaningful oracle test (the score could differ by
  more than the estimator's own inherent variance). Mitigation: if
  Green-time tuning requires a band wider than, say, `0.15-0.20` to pass
  reliably (for EITHER dtype), STOP and flag to the user — a band that wide
  no longer meaningfully tests correctness vs. a bug that happens to still
  land in-range; this is a genuine open numeric question this plan cannot
  resolve without running the actual regen, exactly as SPEC's own
  unresolved-question note anticipated.
- Risk (Plan-Check Pass-1 Issue 2, now resolved): an f32 fixture key
  regenerated but never asserted against, silently leaving the f32 OOB path
  numerically unverified. Mitigation: the f32 Red/Green pair above and the
  explicit Completion Criteria checkbox.

---

## TASK-07 — RF-OOB-01: `oob_score_` sklearn-parity oracle test (regressor)

- **Spec:** `RF-OOB-01`
- **Order:** 7 (Wave 2)
- **Depends on:** TASK-06 (consumes its `gen_oracle.py`/`OOB_MARGIN`
  precedent; extends the regressor's own generator function, still
  sequenced after TASK-06 to avoid a same-file collision)
- **Parallel with:** none

### Objective
Mirrors TASK-06 for `RandomForestRegressor`, using R² instead of accuracy.

### Specification References
- `SPEC-RF-OOB-01` — regressor half.

### Context and Evidence
- Mirrors TASK-06's citations, regressor file.

### Files
- Modify: `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs`
- Modify: `crates/mlrs-algos/tests/random_forest_regressor_test.rs`
- Modify: `scripts/gen_oracle.py`,
  `tests/fixtures/rf_reg_{f32,f64}_seed42.npz`

### TDD Sequence

#### 1. Red
- Test name: `oob_score_within_statistical_band_f64` in
  `random_forest_regressor_test.rs`, mirroring TASK-06.
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test oob_score_within_statistical_band_f64`
- **Companion Red test (Plan-Check Pass-1 Issue 2 — f32, mandatory,
  mirrors TASK-06's f32 addition exactly):** Test name
  `oob_score_within_statistical_band_f32` against `rf_reg_f32_seed42.npz`,
  asserting against a SEPARATE `OOB_MARGIN_F32` constant (regressor's own
  R²-based band, independently tuned from the classifier's
  accuracy-based band — do not reuse TASK-06's `OOB_MARGIN_F32` value
  across estimators). Run: `cargo test -p mlrs-algos --features cpu --test
  random_forest_regressor_test oob_score_within_statistical_band_f32`

#### 2. Green
- Mirror TASK-06's implementation for the regressor, INCLUDING the f32
  fixture regeneration/assertion (both dtype `ref_oob_score` keys asserted,
  not just f64).
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test oob_score_within_statistical_band_f64`
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test oob_score_within_statistical_band_f32`

#### 3. Refactor
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test random_forest_regressor_test`
- Run: `cargo test -p mlrs-algos --features cpu` (full regression — Wave 1/2
  RF-IMP-01/RF-OOB-01 complete after this task)
- Confirm: `ref_oob_score` in the regressor's f32 fixture is asserted
  against (not orphaned — Plan-Check Pass-1 Issue 2, mirrors TASK-06).

### Implementation Steps
1-6. Mirror TASK-06's steps for the regressor, INCLUDING the mandatory f32
     companion Red/Green pair (step 2 in TASK-06's own Implementation
     Steps) and its independently-tuned `OOB_MARGIN_F32`.

### Completion Criteria
- [ ] Same shape as TASK-06 (BOTH f64 and f32 Red/Green pairs, both
      Completion Criteria checkboxes, including the f32-fixture-not-orphaned
      check), regressor variant.

### Risks and Guardrails
- Same as TASK-06, including the Plan-Check Pass-1 Issue 2 f32-coverage
  guardrail.

---

## TASK-08 — PY-ENS-01: `PyRandomForestClassifier` (fit/predict/predict_proba/importances/oob)

- **Spec:** `PY-ENS-01`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 8 (Wave 3)
- **Depends on:** TASK-07 (needs the full RF-IMP-01/RF-OOB-01 Rust surface)
- **Parallel with:** none (creates the shared `ensemble.rs` file TASK-09
  appends to)

### Objective
`crates/mlrs-py/src/estimators/ensemble.rs` exists with `PyRandomForestClassifier`:
`fit(x,y,rows,cols)`, `predict_labels`, `predict_proba_f32/_f64`, `classes_`,
`is_fitted`, `dtype`, `feature_importances_f32/_f64`, `oob_score_f32/_f64`
(returning `PyResult<Option<f32|f64>>`), plus `max_features` string/int/float/None
parsing and the `oob_score` constructor arg.

### Specification References
- `SPEC-PY-ENS-01` — base binding contract.
- `SPEC-RF-IMP-02` — `feature_importances_` PyO3 surface (this task exposes
  the Rust accessor; the Python `@property` wiring is TASK-11).
- `SPEC-RF-OOB-02` — `oob_score_`/`oob_score` PyO3 surface (same split).

### Context and Evidence
- `crates/mlrs-py/src/estimators/naive_bayes.rs:365-418` (`PyGaussianNB::fit`)
  — the exact template, `[VERIFIED: LOCAL — full read, see "Binding-layer
  template" above]`.
- `crates/mlrs-py/src/dispatch.rs:158-185` (`any_estimator_typestate!`) —
  the CORRECT macro (`[VERIFIED: LOCAL — full read]`).
- `crates/mlrs-py/src/estimators/cluster.rs:64-121` (`PyKMeans::fit`) — the
  string-hyperparameter-parse-to-`ValueError` precedent this task's
  `max_features` parser follows (`[VERIFIED: LOCAL — full read]`; SPEC's
  separate citation of a `parse_hdbscan_metric`-style parser in the same
  file is consistent with this pattern, not independently re-verified here
  — confirm the exact `PyValueError::new_err` message format at Green time
  by reading `cluster.rs`'s HDBSCAN metric-parse block directly).
- `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs:80-121`
  (builder defaults) — `n_estimators=100, max_depth=10, n_bins=32,
  max_features=Sqrt, min_samples_split=2.0, min_samples_leaf=1.0,
  bootstrap=true, seed=42`, PLUS `oob_score=false` (TASK-05) —
  `[VERIFIED: LOCAL — full read]`.
- `crates/mlrs-py/src/errors.rs` — `algo_err_to_py`, `build_err_to_py`,
  `not_fitted` (used, not modified — `[VERIFIED: LOCAL — imports confirmed
  in naive_bayes.rs:62]`).

### Files
- Create: `crates/mlrs-py/src/estimators/ensemble.rs`
- Modify: `crates/mlrs-py/src/estimators/mod.rs` (add `pub mod ensemble;`)
- Create: `crates/mlrs-py/tests/test_random_forest.py` (Rust-side PyO3
  not-fitted/dtype-guard tests, per AGENTS.md §2 — tests never in-source)

### TDD Sequence

#### 1. Red
- Test name: `test_predict_before_fit_raises` in
  `crates/mlrs-py/tests/test_random_forest.py`.
- Setup: `mlrs._mlrs.RandomForestClassifier()` (unfit), call
  `.predict_labels(x_capsule, 1, 1)` with any placeholder capsule.
- Expected: raises (the `not_fitted` mapped `PyValueError`).
- Expected initial failure: `ModuleNotFoundError`/`AttributeError` —
  `RandomForestClassifier` is not yet an attribute of `_mlrs` (the crate
  does not compile with this pyclass yet, so `cargo test -p mlrs-py` itself
  fails to build — the Red state IS the crate not compiling, matching the
  "compile error" contract used elsewhere in this plan for a not-yet-created
  type).
- Run: `cargo test -p mlrs-py --features cpu`

#### 2. Green
- Create `ensemble.rs` with a module doc-comment (mirrors `naive_bayes.rs:1-41`'s
  style) and:
  - A `parse_max_features(v: &Bound<'_, PyAny>, n_features: usize) ->
    PyResult<MaxFeatures>` free function (or inline match in `fit`) handling
    `"sqrt"|"log2"` (case-sensitive, sklearn convention) → `Sqrt|Log2`;
    `None` → `All`; a Python `int` → `Value(v as usize)`; a Python `float`
    → `Value((v * n_features as f64).ceil() as usize)` (sklearn's own
    fraction-to-count rule); any other string → `PyValueError` naming the
    bad value.
  - `crate::any_estimator_typestate! { any: AnyRandomForestClassifier, algo:
    mlrs_algos::ensemble::random_forest_classifier::RandomForestClassifier,
    unfit: { n_estimators: usize, max_depth: usize, n_bins: usize,
    max_features: PyObject, min_samples_split: f64, min_samples_leaf: f64,
    bootstrap: bool, oob_score: bool, seed: u64 }, }` (storing the RAW
    `max_features` Python object in the `Unfit` arm and resolving it to
    `MaxFeatures` at `fit()` time, once `cols` — the feature count — is
    known, mirroring the general "defer data-dependent resolution to fit"
    discipline SPEC §4.1 already documents for `MaxFeatures::resolve`).
  - `#[pyclass] struct PyRandomForestClassifier { inner: AnyRandomForestClassifier
    }`, `#[new]` with the `#[pyo3(signature = (n_estimators=100,
    max_depth=10, n_bins=32, max_features="sqrt".into_py(...), ...))]`
    defaults matching the Rust builder defaults verbatim, `fit(x,y,rows,cols)`
    mirroring `PyGaussianNB::fit` exactly (dtype dispatch, `guard_f64()?`,
    `build::<F>().map_err(build_err_to_py)?`, `TypestateFit::fit(...).map_err(algo_err_to_py)?`),
    `predict_labels`, `predict_proba_f32/_f64`, `classes_`, `is_fitted`,
    `dtype` (all thin delegations mirroring `naive_bayes.rs`'s
    `g_labels`/`g_pf32`/etc. free-function pattern), `feature_importances_f32(&self)
    -> PyResult<Vec<f32>>` / `_f64` (call `.feature_importances()` on the
    fitted arm, `.to_vec()`, `not_fitted(...)` on the `Unfit` arm),
    `oob_score_f32(&self) -> PyResult<Option<f32>>` / `_f64` (call
    `.oob_score()`, `not_fitted(...)` on `Unfit`).
- Add `pub mod ensemble;` to `estimators/mod.rs`.
- Run: `cargo test -p mlrs-py --features cpu`

#### 3. Refactor
- Factor the `predict_labels`/`predict_proba_f32/_f64`/`classes_`/`is_fitted`/`dtype`
  delegation into free functions (mirroring `nb_surface_fns!` in
  `naive_bayes.rs:130+`) ONLY if a second consumer (TASK-09's regressor,
  which has a DIFFERENT surface — no `classes_`/`predict_proba`, so no
  macro reuse across classifier/regressor here) would benefit; since
  `PyRandomForestRegressor` (TASK-09) has a materially different method
  set, do NOT force a shared macro across the two — keep the free
  functions PyRandomForestClassifier-specific, matching the simpler
  `PyKMeans`-style single-`#[pymethods]`-block precedent instead of the
  5-variant `naive_bayes.rs` macro (which existed because FIVE NB
  estimators share an IDENTICAL surface — RF classifier has no sibling
  with an identical surface in this file).
- Run: `cargo test -p mlrs-py --features cpu`

#### 4. Verify
- Run: `cargo test -p mlrs-py --features cpu`
- Confirm: `cargo build -p mlrs-py --features cpu` is clean (no leftover
  dead-code warnings on the new module).

### Implementation Steps
1. Write the Red test.
2. Write `test_max_features_bogus_string_raises_value_error`,
   `test_oob_score_true_without_bootstrap_raises_value_error` (the
   `BuildError::OobRequiresBootstrap` mapped through `build_err_to_py`).
3. Implement `ensemble.rs` + `estimators/mod.rs` wiring to pass all Red
   tests.
4. Run the full `mlrs-py` regression suite.

### Completion Criteria
- [ ] `cargo test -p mlrs-py --features cpu` fails to build before Green
      (Red state).
- [ ] Passes after Green.
- [ ] `feature_importances_f32/_f64` and `oob_score_f32/_f64` are present on
      `PyRandomForestClassifier` (RF-IMP-02/RF-OOB-02 Rust half satisfied
      here, not deferred).
- [ ] `max_features` bogus-string and `oob_score`-without-`bootstrap` both
      raise `PyValueError` (not a panic).

### Risks and Guardrails
- Risk (SPEC risk 1): using `any_estimator!` instead of
  `any_estimator_typestate!` — the "WRONG monomorphization" trap. Mitigation:
  the Green step explicitly names the typestate macro; Verify step's
  `cargo test -p mlrs-py --features cpu` compiling IS the guardrail (a
  wrong-macro use fails to compile against `TypestateFit::fit`'s signature).
- Risk: `max_features` stored as a raw `PyObject` in the `Unfit` arm
  requiring the GIL to inspect later — since resolution happens at `fit()`
  time (still holding the GIL, before `py.detach`), this is safe; do NOT
  attempt to resolve `max_features` INSIDE the `py.detach` closure (it
  captures no `Python<'_>` token there by design).

---

## TASK-09 — PY-ENS-02: `PyRandomForestRegressor` (fit/predict/importances/oob)

- **Spec:** `PY-ENS-02`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 9 (Wave 3)
- **Depends on:** TASK-08 (same file, appended after)
- **Parallel with:** none

### Objective
`PyRandomForestRegressor` appended to `ensemble.rs`: `fit`, `predict_f32/_f64`,
`is_fitted`, `dtype`, `feature_importances_f32/_f64`, `oob_score_f32/_f64`,
`max_features` parsing (regressor default `"all"` — sklearn `1.0`, not
`"sqrt"`).

### Specification References
- `SPEC-PY-ENS-02`, `SPEC-RF-IMP-02`, `SPEC-RF-OOB-02` — regressor halves.

### Context and Evidence
- `crates/mlrs-py/python/mlrs/linear.py` (predict-only float accessor
  pattern, `[VERIFIED: LOCAL — read above]`) composed with TASK-08's
  typestate-fit-with-y template — no single existing file has BOTH
  "typestate consuming fit(x,y)" AND "float predict, no proba/classes" in
  one place (SPEC §5.2 already flags this as a composition, not a direct
  template — confirmed true by this Planner's own reads).
- `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` defaults —
  `max_features=All` (`[VERIFIED: CODEGRAPH via RESEARCH.md §5.1]`, not
  independently re-read this pass beyond citation — confirm exact constant
  names at Green time by opening the file, mirroring TASK-08's
  classifier-side constant names).

### Files
- Modify: `crates/mlrs-py/src/estimators/ensemble.rs` (append)
- Modify: `crates/mlrs-py/tests/test_random_forest.py` (append regressor
  cases)

### TDD Sequence

#### 1. Red
- Test name: `test_regressor_predict_before_fit_raises`.
- Run: `cargo test -p mlrs-py --features cpu`

#### 2. Green
- Mirror TASK-08's classifier `#[pyclass]` shape, minus `classes_`/`predict_proba`,
  plus `predict_f32(&self, ...) -> PyResult<Vec<f32>>` / `_f64` (composing
  the typestate `Predict::predict` trait call, mirroring how
  `LogisticRegression::predict_proba`'s device-call shape works but for the
  simpler `Predict` trait instead of `PredictProba`).
- Run: `cargo test -p mlrs-py --features cpu`

#### 3. Refactor
- Run: `cargo test -p mlrs-py --features cpu`

#### 4. Verify
- Run: `cargo test -p mlrs-py --features cpu`

### Implementation Steps
1. Write the Red test + a `max_features` default-is-all-not-sqrt assertion
   test (constructs with no args, fits, confirms no `ValueError` and that
   the resolved feature count equals `n_features` — an indirect check since
   `MaxFeatures` itself is not Python-visible).
2. Implement.
3. Run the full regression suite.

### Completion Criteria
- [ ] Mirrors TASK-08's checklist for the regressor.

### Risks and Guardrails
- Same "wrong monomorphization" guardrail as TASK-08.

---

## TASK-10 — PY-ENS-05 (RF): `lib.rs` + `estimators/mod.rs` registration

- **Spec:** `PY-ENS-05`
- **Order:** 10 (Wave 4a)
- **Depends on:** TASK-09
- **Parallel with:** TASK-11 (disjoint files: `lib.rs` vs. `ensemble.py`)

### Objective
`_mlrs`'s `#[pymodule]` registers `PyRandomForestClassifier`/`PyRandomForestRegressor`
(registration count 32→34); the stale "12 estimator"/"30" doc comments at
`lib.rs:65,178,201` are corrected.

### Specification References
- `SPEC-PY-ENS-05` — registration + stale-comment correction.

### Context and Evidence
- `crates/mlrs-py/src/lib.rs:244-270` (existing `add_class` block,
  `[VERIFIED: LOCAL grep]`), `:65,178,201` (stale comments, `[VERIFIED:
  LOCAL grep]`).

### Files
- Modify: `crates/mlrs-py/src/lib.rs`

### TDD Sequence

#### 1. Red
- Test name: `test_random_forest_classifier_registered_on_mlrs` in
  `crates/mlrs-py/tests/test_random_forest.py`.
- Setup: `import mlrs._mlrs as m; hasattr(m, "RandomForestClassifier")`.
- Expected: `True`.
- Expected initial failure: `False` — not yet registered.
- Run: `cargo test -p mlrs-py --features cpu` (this specific assertion is a
  Python-side pytest, so it actually runs via `pytest
  crates/mlrs-py/python/tests/` post-`maturin develop` — this task's Red
  test is more naturally a Rust-side `PyModule::getattr` check inside
  `crates/mlrs-py/tests/test_random_forest.py`'s existing pyo3-embedded
  harness; use `Python::attach(|py| { let m = py.import("mlrs._mlrs")?;
  assert!(m.hasattr("RandomForestClassifier")?); })` mirroring however the
  existing Rust-side PyO3 test harness in `crates/mlrs-py/tests/` imports
  `_mlrs` today — confirm the exact harness call at Green time by reading
  an existing file in `crates/mlrs-py/tests/`, e.g. `test_naive_bayes.py`.)

#### 2. Green
- Add `use estimators::ensemble::{PyRandomForestClassifier,
  PyRandomForestRegressor};` and two `m.add_class::<...>()?;` calls.
- Correct the "12 estimator"/"30" comments to the current accurate counts.
- Run: `cargo test -p mlrs-py --features cpu`

#### 3. Refactor
- None (mechanical registration).
- Run: `cargo test -p mlrs-py --features cpu`

#### 4. Verify
- Run: `cargo test -p mlrs-py --features cpu`
- Run: `cargo build -p mlrs-py --features cpu` clean.

### Implementation Steps
1. Write the Red test.
2. Register both classes; fix stale comments.
3. Run the full regression suite.

### Completion Criteria
- [ ] Both classes importable from `_mlrs` after Green.
- [ ] No remaining "12 estimator"/"30" stale comment in `lib.rs`.

### Risks and Guardrails
- Risk: forgetting the SECOND `HistGradientBoosting*` pair means this
  count-comment fix must be revisited by TASK-20 too — this task fixes the
  comment to the CORRECT INTERMEDIATE count (32+2=34) only; TASK-20 fixes it
  again to the final count (36). Document this explicitly in the comment
  text itself if the exact wording risks going stale mid-plan (e.g. avoid
  hard-coding "34" in a way TASK-20 must remember to bump — prefer wording
  that names the estimator families rather than a raw count, if the
  existing comment style allows it; otherwise TASK-20 MUST update the same
  line again).

---

## TASK-11 — PY-ENS-01 (Python shim): `ensemble.py` `RandomForestClassifier`

- **Spec:** `PY-ENS-01`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 11 (Wave 4b)
- **Depends on:** TASK-09
- **Parallel with:** TASK-10

### Objective
`crates/mlrs-py/python/mlrs/ensemble.py` exists with `RandomForestClassifier(ClassifierMixin,
MlrsBase)`: `__init__` (defaults matching TASK-08's `#[new]` verbatim),
`fit`, `predict`, `predict_proba`, `feature_importances_` (`@property`,
always present once fitted), `oob_score_` (`@property`, raises
`AttributeError` when `oob_score=False` — sklearn parity, per "Resolved
planning decisions").

### Specification References
- `SPEC-PY-ENS-01`, `SPEC-RF-IMP-02`, `SPEC-RF-OOB-02`.

### Context and Evidence
- `crates/mlrs-py/python/mlrs/naive_bayes.py:64-70` (`GaussianNB.fit`,
  `[VERIFIED: LOCAL — full read]`) — the exact `_normalize`/`_normalize_y`/`_store_fit`
  template.
- `crates/mlrs-py/python/mlrs/base.py` — `MlrsBase`/`ClassifierMixin` (not
  independently re-read this pass beyond SPEC's citation of
  `:32-183`/`[VERIFIED: LOCAL]` — Green time confirms `_suffixed`/`_to_output`/`_check_fitted`
  exact signatures).
- The "Resolved planning decisions" `AttributeError`-vs-`NotFittedError`
  distinction for `oob_score_`.

### Files
- Create: `crates/mlrs-py/python/mlrs/ensemble.py`

### TDD Sequence

#### 1. Red
- Test name (Python, requires the built wheel — run via `pytest`, not
  `cargo test`): `test_random_forest_classifier_importable` in a SCRATCH
  location first, folded into TASK-14's `test_oracle_ensemble.py` for the
  real assertions; THIS task's own Red proof is structural, checked via the
  Rust-side `cargo test -p mlrs-py` build NOT being affected (a pure-Python
  file has no `cargo test` Red state) — instead, this task's Red/Green is
  validated via `crates/mlrs-py/python/tests/test_shims.py`'s EXISTING
  `test_all_shims_importable` (parametrized over `ALL_SHIMS`, auto-derived
  from `mlrs.__all__`) — but `RandomForestClassifier` is not YET in
  `mlrs.__all__` (that is TASK-13), so THIS task's Red state is simply "the
  file/class does not exist yet" and Green is "the file/class exists,
  importable via `from mlrs.ensemble import RandomForestClassifier`
  directly" (bypassing `__all__` for this task's own narrow scope) — assert
  via a minimal ad-hoc script or defer the FIRST real pytest assertion to
  TASK-14 once `__init__.py` wiring (TASK-13) makes it reachable via
  `mlrs.RandomForestClassifier`. Document this explicitly rather than
  inventing a premature pytest file this task would immediately obsolete.
- Practical Red proof for THIS task: `python3 -c "from mlrs.ensemble import
  RandomForestClassifier"` fails with `ModuleNotFoundError` before Green.

#### 2. Green
- Implement `RandomForestClassifier(ClassifierMixin, MlrsBase)` per the
  contract sketch in SPEC §4.3, mirroring `naive_bayes.py`'s `fit`:
  `_normalize(X)` → `_normalize_y(y, dtype=...)` → `self._ext().RandomForestClassifier(...)`
  → `.fit(xa, ya, rows, cols)` → `self._store_fit(obj, cols)` →
  `self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)`.
  `predict`/`predict_proba` mirror `naive_bayes.py`'s equivalents.
  `feature_importances_` property: `self._check_fitted(); return
  self._to_output(self._suffixed("feature_importances_")(), (-1,), None,
  self._np_float())` (mirrors `coef_`'s shape in `linear.py:234-237`).
  `oob_score_` property: `self._check_fitted()` (raises `NotFittedError` if
  unfit — standard); THEN, if fitted, call the `_mlrs` accessor which
  returns `Optional[float]`; if it returns `None` (i.e. `self.oob_score is
  False`), `raise AttributeError("'RandomForestClassifier' object has no
  attribute 'oob_score_' (oob_score=False)")` in the PYTHON shim layer (per
  "Resolved planning decisions" — the None→AttributeError translation
  happens in the Python shim, not the PyO3 layer, since PyO3 has no
  precedent for raising `AttributeError` from a method call and the shim
  layer already owns every other attribute-presence decision).
- Practical Red proof passes: `python3 -c "from mlrs.ensemble import
  RandomForestClassifier"` succeeds.

#### 3. Refactor
- None significant — one class, following an established template exactly.

#### 4. Verify
- `python3 -c "from mlrs.ensemble import RandomForestClassifier;
  RandomForestClassifier()"` constructs without error (no compiled
  extension needed for construction, per `naive_bayes.py`'s pre-build
  importability precedent).

### Implementation Steps
1. Confirm the Red (import) failure.
2. Implement `RandomForestClassifier`.
3. Confirm construction succeeds pre-build (no `_mlrs` import needed until
   `.fit()` is called).

### Completion Criteria
- [ ] `RandomForestClassifier` importable and zero-arg-constructible from
      `mlrs.ensemble` before any wheel is built.
- [ ] `__init__` stores every ctor arg verbatim under its sklearn name
      (purity rule, matches `LogisticRegression.__init__`'s style).
- [ ] `oob_score_`'s `AttributeError`-vs-`NotFittedError` distinction is
      implemented exactly as resolved above.

### Risks and Guardrails
- Risk: `feature_importances_`/`oob_score_` accessor names on the `_mlrs`
  object not matching TASK-08's actual method names
  (`feature_importances_f32`/`_f64` vs. a bare `feature_importances_`) —
  the `_suffixed(...)` helper (used identically for `coef_`/`intercept_` in
  `linear.py`) already handles the `_f32`/`_f64` dtype-suffix dispatch, so
  this task's Green step MUST call `_suffixed("feature_importances_")` (NOT
  a raw method name), matching `linear.py:234-237`'s exact pattern.

---

## TASK-12 — PY-ENS-02 (Python shim): `ensemble.py` `RandomForestRegressor`

- **Spec:** `PY-ENS-02`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 12 (Wave 4b)
- **Depends on:** TASK-11 (same file, appended after)
- **Parallel with:** none

### Objective
`RandomForestRegressor(RegressorMixin, MlrsBase)` appended to `ensemble.py`,
mirroring TASK-11 minus `predict_proba`/`classes_`.

### Specification References
- `SPEC-PY-ENS-02`, `SPEC-RF-IMP-02`, `SPEC-RF-OOB-02`.

### Context and Evidence
- `crates/mlrs-py/python/mlrs/neighbors.py:85-104`
  (`KNeighborsRegressor`, `[VERIFIED: LOCAL — full read]`) — the
  `RegressorMixin`/predict-only shim shape this task mirrors.

### Files
- Modify: `crates/mlrs-py/python/mlrs/ensemble.py` (append)

### TDD Sequence

#### 1. Red
- `python3 -c "from mlrs.ensemble import RandomForestRegressor"` fails
  before Green.

#### 2. Green
- Implement, mirroring TASK-11 minus `classes_`/`predict_proba`.

#### 3. Refactor
- None.

#### 4. Verify
- `python3 -c "from mlrs.ensemble import RandomForestRegressor;
  RandomForestRegressor()"` succeeds.

### Implementation Steps
1-3. Mirror TASK-11.

### Completion Criteria
- [ ] Mirrors TASK-11's checklist for the regressor.

### Risks and Guardrails
- Same `_suffixed(...)` risk as TASK-11.

---

## TASK-13 — PY-ENS-05 (RF): `__init__.py` wiring

- **Spec:** `PY-ENS-05`
- **Order:** 13 (Wave 5)
- **Depends on:** TASK-10, TASK-12
- **Parallel with:** none

### Objective
`mlrs.RandomForestClassifier`/`mlrs.RandomForestRegressor` importable from
the top-level `mlrs` package; both appear in `__all__` (which
auto-registers them into `test_shims.py::ALL_SHIMS`, per the "Binding-layer
template" note above — no manual edit needed there).

### Specification References
- `SPEC-PY-ENS-05`.

### Context and Evidence
- `crates/mlrs-py/python/mlrs/__init__.py:20-98` — the existing
  `from .family import (...)` + `__all__` list structure, `[VERIFIED: LOCAL
  — full read]`.

### Files
- Modify: `crates/mlrs-py/python/mlrs/__init__.py`

### TDD Sequence

#### 1. Red
- Test name: `test_random_forest_classifier_top_level_import` — a NEW
  minimal pytest in `crates/mlrs-py/python/tests/test_shims.py` (or reuse
  the auto-derived `ALL_SHIMS` machinery, which will pick it up once
  `__all__` is updated — so this task's OWN Red proof is simply: `import
  mlrs; mlrs.RandomForestClassifier` raises `AttributeError` before Green).
- Run: `python3 -c "import mlrs; mlrs.RandomForestClassifier"` (expect
  `AttributeError` before Green).

#### 2. Green
- Add `from .ensemble import RandomForestClassifier, RandomForestRegressor`
  and both names to `__all__`.
- Run: `python3 -c "import mlrs; mlrs.RandomForestClassifier;
  mlrs.RandomForestRegressor"` (expect success).

#### 3. Refactor
- None.

#### 4. Verify
- `pytest crates/mlrs-py/python/tests/test_shims.py -k
  test_all_shims_importable` — now covers both new names automatically
  (pre-build, no wheel needed for THIS specific test since it only imports
  the pure-Python shim class, per `naive_bayes.py`'s precedent).

### Implementation Steps
1. Confirm the Red (`AttributeError`) state.
2. Add the import + `__all__` entries.
3. Run `test_shims.py::test_all_shims_importable` (now auto-covers the two
   new names) and `test_family_mixins_composed`-style assertions if
   applicable.

### Completion Criteria
- [ ] Both names top-level importable.
- [ ] `test_shims.py::ALL_SHIMS` now includes both (verified by re-running
      `test_all_shims_importable` and `test_fit_returns_self_signature`).

### Risks and Guardrails
- Risk: `test_shims.py::test_fitted_attr_raises_before_fit`'s MANUAL list
  does NOT auto-cover the new `feature_importances_` entries — this task
  does not add those (TASK-16 does); confirm this task does not
  accidentally break OTHER tests by omission (it will not — an absent
  parametrize entry is simply untested, not a failure).

---

## TASK-14 — PY-ENS-01: Python oracle replay (RandomForestClassifier)

- **Spec:** `PY-ENS-01`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 14 (Wave 6)
- **Depends on:** TASK-13 (needs a built wheel: `maturin develop -m
  crates/mlrs-py/pyproject/cpu.pyproject.toml`)
- **Parallel with:** none (owns `test_oracle_ensemble.py`, TASK-15 appends)

### Objective
`crates/mlrs-py/python/tests/test_oracle_ensemble.py` replays
`rf_cls_{f32,f64}_seed42.npz` (now carrying `ref_feature_importances` and
`ref_oob_score` per TASK-02/TASK-06) through the full Python path:
deterministic-tier exact predict/proba, statistical-tier accuracy band,
`feature_importances_` (`atol=0.05` on deterministic tier — **[RESOLVED at
TASK-02 Green time: not an exact match, see TASK-02's Objective]** —
+ qualitative ranking as the primary signal),
`oob_score_` (statistical band + `AttributeError` when `oob_score=False`),
`ValueError` on `oob_score=True, bootstrap=False`.

### Specification References
- `SPEC-PY-ENS-01`, `SPEC-RF-IMP-02`, `SPEC-RF-OOB-02` — full acceptance
  scenario 1, 6, 7, 8 (SPEC §6).

### Context and Evidence
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-70` — the
  `_atol(fixture)` dtype-branch, `@requires_f64` marker, `np.load` fixture
  harness template, `[VERIFIED: LOCAL — cited by SPEC.md, RESEARCH.md §5.4]`.
- Fixture keys per SPEC §3: `X, y, Xq, yq, det_pred_train, det_proba_train,
  stat_acc_test` + this plan's additions `ref_feature_importances`,
  `ref_oob_score`.

### Files
- Create: `crates/mlrs-py/python/tests/test_oracle_ensemble.py`

### TDD Sequence

#### 1. Red
- Test name: `test_random_forest_classifier_deterministic`.
- Setup: build `mlrs.RandomForestClassifier(bootstrap=False,
  max_features=None, max_depth=12, n_estimators=2, ...)` (mirroring the
  Rust deterministic-tier construction) on `X`/`y` from the fixture.
- Expected: `.fit(X,y).predict(X)` matches `det_pred_train` exactly;
  `.predict_proba(X)` matches `det_proba_train` within `1e-5`.
- Expected initial failure: `ModuleNotFoundError` for `_mlrs` (pre-build) or
  a real assertion failure if the wheel exists but the estimator was never
  built correctly — either way, this is the FIRST test in the file to
  actually exercise the compiled extension, so it is the true Red/Green
  boundary for the whole PY-ENS-01 Rust+Python integration.
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml &&
  pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -k
  test_random_forest_classifier_deterministic`

#### 2. Green
- (No production code changes expected here if TASK-08..13 are correct —
  this task's Green step is discovering and fixing any integration-only
  bug the isolated Rust/Python unit tests did not catch, e.g. a
  hyperparameter-name mismatch between the Python `__init__` and the Rust
  `#[new]` signature.)
- Run the same command.

#### 3. Refactor
- None expected.

#### 4. Verify
- Run the full new test file.
- Run `pytest crates/mlrs-py/python/tests/` (full regression — confirms no
  existing shim broke).

### Implementation Steps
1. Write `test_random_forest_classifier_deterministic`.
2. Write `test_random_forest_classifier_statistical` (accuracy within
   `ACC_MARGIN` band on `Xq`/`yq` vs `stat_acc_test`).
3. Write `test_random_forest_classifier_max_features_invalid_raises`.
4. Write `test_random_forest_classifier_not_fitted_raises`.
5. Write `test_random_forest_classifier_feature_importances_close` (**[RESOLVED
   at TASK-02 Green time, 2026-07-18: `atol=0.05`, NOT exact `1e-5`]** —
   sklearn's own splitter tie-break nondeterminism means no tier supports an
   exact match for `feature_importances_`, see TASK-02's Objective — against
   `ref_feature_importances`, deterministic-tier construction) +
   `test_random_forest_classifier_feature_importances_sums_to_one`.
6. Write `test_random_forest_classifier_oob_score_statistical_band` (uses
   `OOB_MARGIN` from TASK-06, duplicated as a Python-side constant with a
   comment cross-referencing the Rust test) +
   `test_random_forest_classifier_oob_score_false_raises_attribute_error`.
7. Write `test_random_forest_classifier_oob_true_bootstrap_false_raises_value_error`.
8. Add `@requires_f64` gating on every f64-dtype variant, mirroring
   `test_oracle_neighbors.py`.
9. Run the full test file + the full `pytest` suite.

### Completion Criteria
- [ ] Every Given/When/Then in SPEC §5 PY-ENS-01/RF-IMP-02/RF-OOB-02 has a
      corresponding passing test.
- [ ] `pytest crates/mlrs-py/python/tests/` full suite green.

### Risks and Guardrails
- Risk: the Rust `OOB_MARGIN` (TASK-06) and this task's Python-side replica
  drift if one is tuned without the other. Mitigation: cite the Rust test
  file/line in a comment here so a future tuning pass finds both.

---

## TASK-15 — PY-ENS-02: Python oracle replay (RandomForestRegressor)

- **Spec:** `PY-ENS-02`, `RF-IMP-02`, `RF-OOB-02`
- **Order:** 15 (Wave 6)
- **Depends on:** TASK-14 (same file, appended after)
- **Parallel with:** none

### Objective
Mirrors TASK-14 for the regressor (R²/error band instead of accuracy).

### Specification References
- `SPEC-PY-ENS-02`, `SPEC-RF-IMP-02`, `SPEC-RF-OOB-02` — scenario 2, 6, 7.

### Context and Evidence
- Mirrors TASK-14's citations, regressor fixture (`rf_reg_*.npz`).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_ensemble.py` (append)

### TDD Sequence
Mirrors TASK-14's four-step structure exactly, regressor variant (R²/MSE
band from the fixture's own documented statistical tier, per SPEC §5
PY-ENS-02).

### Implementation Steps
1-9. Mirror TASK-14, substituting `predict` for `predict`/`predict_proba`
     and R² for accuracy.

### Completion Criteria
- [ ] Mirrors TASK-14's checklist for the regressor.
- [ ] `pytest crates/mlrs-py/python/tests/` full suite green (Wave 3-6 RF
      binding work complete after this task).

### Risks and Guardrails
- Same as TASK-14.

---

## TASK-16 — PY-ENS-05 (RF): gate-test updates

- **Spec:** `PY-ENS-05`
- **Order:** 16 (Wave 7)
- **Depends on:** TASK-15
- **Parallel with:** none

### Objective
`test_params.py::EXPECTED_PARAMS` gains `RandomForestClassifier`/`RandomForestRegressor`
entries; `test_shims.py` gains `feature_importances_` to the generic
`test_fitted_attr_raises_before_fit` list (both estimators) PLUS a NEW
dedicated test for `oob_score_`'s conditional-`AttributeError` behavior
(per "Resolved planning decisions"); `test_estimator_checks.py::_estimators()`
gains both instances, triaged against the `check_estimator` sweep (pass or
documented xfail).

### Specification References
- `SPEC-PY-ENS-05` — scenario 5 (SPEC §6).

### Context and Evidence
- `test_params.py:29+` (`EXPECTED_PARAMS` dict shape, `[VERIFIED: LOCAL]`).
- `test_shims.py:178-206` (`test_fitted_attr_raises_before_fit`,
  `[VERIFIED: LOCAL — full read]`).
- `test_estimator_checks.py:33-90+` (`_estimators()` list + the
  by-design-unsupported xfail dict shape, `[VERIFIED: LOCAL]`).

### Files
- Modify: `crates/mlrs-py/python/tests/test_params.py`
- Modify: `crates/mlrs-py/python/tests/test_shims.py`
- Modify: `crates/mlrs-py/python/tests/test_estimator_checks.py`

### TDD Sequence

#### 1. Red
- Test name: `test_default_params_match_sklearn_names["RandomForestClassifier"]`
  (parametrized — the test ALREADY exists and iterates `EXPECTED_PARAMS`
  keys; adding a new key without a matching dict entry does nothing, so
  this task's Red state is the INVERSE — add the estimator to
  `mlrs.__all__`-derived coverage first (already done, TASK-13), then the
  Red proof is: running `test_default_params_match_sklearn_names` with a
  hand-added `"RandomForestClassifier": {}` (empty/wrong dict) FAILS
  against the real `get_params()` output, proving the test harness
  correctly detects a mismatch before the real dict is filled in).
- Run: `pytest crates/mlrs-py/python/tests/test_params.py -k RandomForest`

#### 2. Green
- Add the two real `EXPECTED_PARAMS` entries (every ctor arg + its
  documented default, matching TASK-08/09/11/12's `__init__` signatures
  verbatim — including `oob_score: False`).
- Add `("RandomForestClassifier", "feature_importances_")` and
  `("RandomForestRegressor", "feature_importances_")` to
  `test_fitted_attr_raises_before_fit`'s parametrize list.
- Add a new test function `test_random_forest_oob_score_conditional_attribute`:
  `mlrs.RandomForestRegressor(oob_score=True).oob_score_` before fit raises
  `NotFittedError`; `mlrs.RandomForestRegressor(oob_score=False).fit(...).oob_score_`
  raises `AttributeError` (needs a tiny valid `X`/`y` — reuse a fixture-free
  hand-built array, mirroring how other `test_shims.py` fitted tests avoid
  needing real oracle fixtures where possible, OR fall back to
  `pytest.importorskip("mlrs")`-gated if it needs a real fit — confirm
  which at Green time).
- Append `mlrs.RandomForestClassifier(n_estimators=5, max_depth=3),
  mlrs.RandomForestRegressor(n_estimators=5, max_depth=3)` (small, cheap
  hyperparameters for the check-sweep's tiny fixtures) to
  `_estimators()` in `test_estimator_checks.py`.
- Run `parametrize_with_checks` locally (post-`maturin develop`) and record
  ANY newly-failing check as a documented `expected_failed_checks` xfail
  entry with a reason, mirroring the existing entries — do NOT assume it
  passes cleanly (SPEC's own explicit "treat as a verification task, not
  assume pass" instruction).
- Run: `pytest crates/mlrs-py/python/tests/test_params.py
  crates/mlrs-py/python/tests/test_shims.py -k RandomForest`

#### 3. Refactor
- None expected.

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/` (full suite).
- Run: `pytest crates/mlrs-py/python/tests/test_estimator_checks.py` (needs
  the built wheel; confirm every RF check either passes or is xfailed with
  a documented reason, never silently skipped).

### Implementation Steps
1. Add the `EXPECTED_PARAMS` entries.
2. Add the `feature_importances_` parametrize entries.
3. Add the dedicated `oob_score_` conditional-attribute test.
4. Add the two `_estimators()` instances; run the check sweep; triage any
   new failure.
5. Run the full `pytest` suite.

### Completion Criteria
- [ ] All three gate files updated; full `pytest` suite green (or every
      failure is a documented, reasoned xfail — never an un-triaged
      failure).
- [ ] `oob_score_`'s conditional-attribute behavior has its own dedicated,
      passing test (not folded into the generic list, per the resolved
      decision above).

### Risks and Guardrails
- Risk: `check_estimator` failing for a reason NOT yet in the existing
  xfail taxonomy (e.g. something specific to RF's `bootstrap`/`oob_score`
  cross-field validation confusing sklearn's generic constructor-check).
  Mitigation: this task's Green step explicitly runs the sweep and reads
  the ACTUAL failure before writing any xfail entry — no xfail is added
  speculatively.

---

## TASK-17 — HGB fixture-freshness gate checkpoint

- **Spec:** locked decision (SPEC frontmatter), gates `SPEC-PY-ENS-03`,
  `SPEC-PY-ENS-04`'s oracle-tolerance-finalization sub-task only.
- **Order:** 17 (Wave 8 — no code dependency, may run any time; placed here
  for narrative clarity)
- **Depends on:** none
- **Parallel with:** Waves 1-7 (inspects `git status` only, touches no
  source file)

### Objective
An explicit, standalone checkpoint (not folded into TASK-18/19/24's Green
step, per the mission's explicit instruction) recording whether
`crates/mlrs-backend/src/prims/hist_gradient_boosting.rs`,
`crates/mlrs-kernels/src/gbt.rs`, `scripts/gen_oracle.py`, and all four
`tests/fixtures/hgb_{cls,reg}_{f32,f64}_seed42.npz` are clean in `git
status --short`. TASK-24 (the ONLY task this gates) may proceed to PIN real
sklearn-comparison tolerances ONLY if this task's re-run (at TASK-24's OWN
Green time, not this task's) shows clean; otherwise TASK-24 implements the
documented skip/xfail mechanism instead.

### Specification References
- `SPEC-PY-ENS-03`, `SPEC-PY-ENS-04` — the blocking precondition (SPEC §5).
- Locked decision (SPEC frontmatter, `locked_decisions` #2/#5).

### Context and Evidence
- `git status --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
  crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py
  tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz
  tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz`
  — run FRESH by this Planner (2026-07-17): **all seven paths show `M`
  (modified, uncommitted)**. `[VERIFIED: LOCAL — command output captured
  verbatim in this plan's "Resolved planning decisions" §Q-HGB-fixture-freshness]`

### Files
- None modified — this is a verification-only task. (No `Create`/`Modify`
  entries; its "artifact" is the recorded finding above and the
  gate re-check instruction TASK-24 must follow.)

### TDD Sequence

This task has no Red/Green/Refactor in the code sense (there is no
production code to test) — it is a **verification checkpoint**. Its
"test" is the `git status --short` command itself, and its "pass/fail" is
binary (clean/dirty), re-run at TASK-24's own Green time as the actual
gating check.

#### 1. Red / 2. Green (checkpoint form)
- Run: `git status --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
  crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py
  tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz
  tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz`
- If output is EMPTY (clean): TASK-24 is UNBLOCKED — proceed with real
  sklearn-pinned tolerances.
- If output is NON-EMPTY (dirty, the state as of THIS plan's writing):
  TASK-24 is BLOCKED for tolerance-pinning — implement the documented
  skip/xfail mechanism (SPEC §5 PY-ENS-03: "Planner must decide the exact
  test-suite-green mechanism; do not silently pin against dirty
  fixtures") — this Planner's decision: use
  `@pytest.mark.xfail(reason="HGB algos churn in flight — see
  .planning/plans/py-ensemble/PLAN.md TASK-17; hist_gradient_boosting.rs/gbt.rs/gen_oracle.py/hgb_*.npz
  uncommitted as of <date>", strict=False)` on the deterministic-tier exact-match
  assertions ONLY (the statistical-tier band assertions, which tolerate
  noise by design, MAY still run un-xfailed if they happen to pass — do not
  blanket-xfail the whole file).

#### 3. Refactor / 4. Verify
- N/A (checkpoint task).

### Implementation Steps
1. Run the `git status --short` command (already run once by this Planner,
   result recorded above as of 2026-07-17 — **dirty**).
2. Record the finding in this task's own execution log at implementation
   time (re-run, do not reuse this plan's stale snapshot).
3. Pass the finding to TASK-24.

### Completion Criteria
- [ ] The `git status --short` command has been run (not assumed) at
      TASK-17's own execution time, and its result (clean/dirty) is
      recorded and handed to TASK-24.

### Risks and Guardrails
- Risk: TASK-24 is executed long after TASK-17, and the state has changed
  (either direction) — mitigation: TASK-24's own Green step explicitly
  RE-RUNS the same `git status --short` command rather than trusting
  TASK-17's snapshot, exactly as this plan's "Resolved planning decisions"
  section already states.

---

## TASK-18 — PY-ENS-03: `PyHistGradientBoostingClassifier` (structural)

- **Spec:** `PY-ENS-03`
- **Order:** 18 (Wave 9)
- **Depends on:** TASK-09 (file state — `ensemble.rs` must have both RF
  classes already; NOT gated by TASK-17, per SPEC §5's explicit
  "not blocking for writing the `#[pyclass]`/shim structure itself")
- **Parallel with:** none (same file as TASK-08/09)

### Objective
`PyHistGradientBoostingClassifier` appended to `ensemble.rs`: `fit`,
`predict_labels`, `predict_proba_f32/_f64`, `classes_`, `is_fitted`,
`dtype` — NO `feature_importances_`/`oob_score_` (not applicable, SPEC §2).
Mechanically identical to TASK-08 minus the RF-only accessors and minus
`max_features`/`bootstrap`/`oob_score` (HGB has none of these).

### Specification References
- `SPEC-PY-ENS-03` — base binding contract (structural half; the
  oracle-tolerance half is TASK-24, gated).

### Context and Evidence
- `crates/mlrs-algos/src/ensemble/hist_gradient_boosting_classifier.rs`
  defaults — `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
  l2_regularization=0.0, min_samples_leaf=20`
  (`[VERIFIED: CODEGRAPH via RESEARCH.md §5.1 and research.md §b]`, not
  independently re-read this pass — confirm exact constant/builder-setter
  names at Green time by opening the file, mirroring TASK-08's discipline).
- TASK-08's `PyRandomForestClassifier` (mirrors its `fit`/typestate
  structure exactly, no `max_features` parsing needed).

### Files
- Modify: `crates/mlrs-py/src/estimators/ensemble.rs` (append)
- Modify: `crates/mlrs-py/tests/test_random_forest.py` → consider renaming
  to a shared `test_ensemble.py` at this point since it now covers HGB too
  (optional cosmetic — if renamed, do it as a distinct step so `git diff`
  stays reviewable, not folded silently into this task's functional
  change).

### TDD Sequence

#### 1. Red
- Test name: `test_hgb_classifier_predict_before_fit_raises`.
- Run: `cargo test -p mlrs-py --features cpu`

#### 2. Green
- Implement `PyHistGradientBoostingClassifier`, mirroring TASK-08's
  `fit`/dtype-dispatch/error-mapping shape with HGB's own builder setters
  (`max_iter, learning_rate, max_depth, n_bins, l2_regularization,
  min_samples_leaf`) and `n_bins` defaulting to `64` (NOT `255` — the
  Python-visible default must match the Rust builder default; the
  `n_bins=255` deterministic-tier override is a TEST-TIME construction
  argument, not a changed default, per SPEC §5 PY-ENS-03's explicit note).
- Run: `cargo test -p mlrs-py --features cpu`

#### 3. Refactor
- Run: `cargo test -p mlrs-py --features cpu`

#### 4. Verify
- Run: `cargo test -p mlrs-py --features cpu`

### Implementation Steps
1. Write the Red test.
2. Implement.
3. Run the full regression suite.

### Completion Criteria
- [ ] `PyHistGradientBoostingClassifier` compiles and passes its not-fitted
      guard test.
- [ ] No `feature_importances_f32/_f64`/`oob_score_f32/_f64` methods exist
      on this class (explicit non-goal — a stray copy-paste from TASK-08
      would be a real defect here).

### Risks and Guardrails
- Risk: copy-pasting TASK-08's RF-specific accessors onto the HGB class by
  mistake. Mitigation: this task's Completion Criteria explicitly checks
  for their ABSENCE.

---

## TASK-19 — PY-ENS-04: `PyHistGradientBoostingRegressor` (structural)

- **Spec:** `PY-ENS-04`
- **Order:** 19 (Wave 9)
- **Depends on:** TASK-18 (same file, appended after)
- **Parallel with:** none

### Objective
Mirrors TASK-18 for the regressor: `fit`, `predict_f32/_f64`, `is_fitted`,
`dtype`.

### Specification References
- `SPEC-PY-ENS-04`.

### Context and Evidence
- Mirrors TASK-18/TASK-09's citations.

### Files
- Modify: `crates/mlrs-py/src/estimators/ensemble.rs` (append)

### TDD Sequence
Mirrors TASK-18's four-step structure, regressor variant.

### Implementation Steps
1-3. Mirror TASK-18.

### Completion Criteria
- [ ] Mirrors TASK-18's checklist, regressor variant.
- [ ] `cargo test -p mlrs-py --features cpu` green (Wave 9 complete).

### Risks and Guardrails
- Same as TASK-18.

---

## TASK-20 — PY-ENS-05 (HGB): `lib.rs` registration

- **Spec:** `PY-ENS-05`
- **Order:** 20 (Wave 10a)
- **Depends on:** TASK-19
- **Parallel with:** TASK-21

### Objective
`PyHistGradientBoostingClassifier`/`Regressor` registered (count 34→36);
final correction of the "12"/"30" stale comments to their true final state.

### Specification References
- `SPEC-PY-ENS-05`.

### Files
- Modify: `crates/mlrs-py/src/lib.rs`

### TDD Sequence
Mirrors TASK-10 exactly, HGB pair.

### Implementation Steps
1-3. Mirror TASK-10.

### Completion Criteria
- [ ] Both HGB classes importable from `_mlrs`.
- [ ] Registration count is now 36 and the doc comments reflect it
      accurately (this is the FINAL correction — no further estimator
      additions are in this plan's scope).

### Risks and Guardrails
- None beyond TASK-10's.

---

## TASK-21 — PY-ENS-03 (Python shim): `ensemble.py` `HistGradientBoostingClassifier`

- **Spec:** `PY-ENS-03`
- **Order:** 21 (Wave 10b)
- **Depends on:** TASK-19
- **Parallel with:** TASK-20

### Objective
`HistGradientBoostingClassifier(ClassifierMixin, MlrsBase)` appended to
`ensemble.py`, mirroring TASK-11 minus `feature_importances_`/`oob_score_`.

### Specification References
- `SPEC-PY-ENS-03`.

### Files
- Modify: `crates/mlrs-py/python/mlrs/ensemble.py` (append)

### TDD Sequence
Mirrors TASK-11's structure minus the two RF-only properties.

### Implementation Steps
1-3. Mirror TASK-11.

### Completion Criteria
- [ ] Importable, zero-arg-constructible, `n_bins` default `64` (Python
      `__init__` matches TASK-18's Rust `#[new]` default verbatim).

### Risks and Guardrails
- Same `_suffixed(...)` risk as TASK-11 (n/a here since no
  `feature_importances_`/`oob_score_` — lower risk than TASK-11/12).

---

## TASK-22 — PY-ENS-04 (Python shim): `ensemble.py` `HistGradientBoostingRegressor`

- **Spec:** `PY-ENS-04`
- **Order:** 22 (Wave 10b)
- **Depends on:** TASK-21 (same file, appended after)
- **Parallel with:** none

### Objective
Mirrors TASK-21 for the regressor.

### Files
- Modify: `crates/mlrs-py/python/mlrs/ensemble.py` (append)

### TDD Sequence / Implementation Steps / Completion Criteria
Mirror TASK-21, regressor variant (no `classes_`/`predict_proba`, plain
`predict`).

### Risks and Guardrails
- Same as TASK-21.

---

## TASK-23 — PY-ENS-05 (HGB): `__init__.py` wiring

- **Spec:** `PY-ENS-05`
- **Order:** 23 (Wave 11)
- **Depends on:** TASK-20, TASK-22
- **Parallel with:** none

### Objective
`mlrs.HistGradientBoostingClassifier`/`Regressor` importable top-level;
`__all__` updated (auto-registers into `test_shims.py::ALL_SHIMS`).

### Files
- Modify: `crates/mlrs-py/python/mlrs/__init__.py`

### TDD Sequence / Implementation Steps / Completion Criteria
Mirrors TASK-13 exactly, HGB pair. All four PY-ENSEMBLE estimators are
top-level-importable after this task.

### Risks and Guardrails
- None beyond TASK-13's.

---

## TASK-24 — PY-ENS-03/04: Python oracle replay (HGB), GATED

- **Spec:** `PY-ENS-03`, `PY-ENS-04`
- **Order:** 24 (Wave 12)
- **Depends on:** TASK-23, TASK-17 (re-checked at THIS task's own Green
  time, not TASK-17's)
- **Parallel with:** none

### Objective
`test_oracle_ensemble.py` (appended) replays `hgb_{cls,reg}_{f32,f64}_seed42.npz`
through the full Python path — deterministic tier (`n_bins=255` explicit
override), statistical tier (defaults) — **conditionally pinned or
xfailed** based on a FRESH `git status --short` re-check of the four named
HGB files at this task's own execution time (per TASK-17's finding AND
guardrail).

### Specification References
- `SPEC-PY-ENS-03`, `SPEC-PY-ENS-04` — scenario 3, 4 (SPEC §6), the
  explicitly gated ones.

### Context and Evidence
- TASK-17's finding (dirty as of 2026-07-17; MUST be re-verified now, not
  assumed).
- `crates/mlrs-algos/tests/hist_gradient_boosting_classifier_test.rs:100-190`
  — the Rust-side deterministic-tier construction (`n_bins(255)`) this
  task's Python construction mirrors (`[VERIFIED: LOCAL — cited by
  research.md §b]`, not independently re-read this pass — confirm exact
  fixture key names at Green time).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_ensemble.py` (append)

### TDD Sequence

#### 1. Red
- Test name: `test_hgb_classifier_deterministic`.
- **Step 0 (mandatory, before writing any assertion):** run `git status
  --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
  crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py tests/fixtures/hgb_*.npz`
  fresh. Branch on the result:
  - **Clean:** proceed as a normal pinned-tolerance Red/Green test,
    mirroring TASK-14's structure exactly (deterministic exact-match +
    statistical band + not-fitted + invalid-input error cases).
  - **Dirty (the state as of this plan's writing):** write the test WITH
    `@pytest.mark.xfail(reason="...", strict=False)` on the
    deterministic-tier exact-match assertion ONLY, per TASK-17's documented
    mechanism; the statistical-tier band assertion and the structural
    assertions (not-fitted, invalid `n_bins`, etc.) are NOT xfailed (they do
    not depend on exact fixture freshness).
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml &&
  pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -k
  test_hgb_classifier_deterministic`

#### 2. Green
- If clean: fix any integration-only bug the isolated unit tests missed.
- If dirty: confirm the `xfail` marker correctly reports `XFAIL` (not
  `XPASS`, which would indicate the fixture accidentally already matches —
  if `XPASS` occurs, this is a SIGNAL the churn may have settled sooner
  than expected; do not treat `XPASS` as success, re-run the `git status`
  check and reconsider un-xfailing).

#### 3. Refactor
- None expected.

#### 4. Verify
- Run the full new test additions.
- Run `pytest crates/mlrs-py/python/tests/` (full regression).

### Implementation Steps
1. Re-run the `git status --short` gate check (Step 0 above) — MANDATORY,
   do not reuse TASK-17's stale snapshot.
2. Branch: pinned tests (clean) or xfailed deterministic-tier + un-xfailed
   statistical-tier tests (dirty).
3. Write `test_hgb_classifier_statistical`,
   `test_hgb_classifier_not_fitted_raises`,
   `test_hgb_classifier_invalid_input_raises`.
4. Mirror all of the above for `test_hgb_regressor_*`.
5. Run the full `pytest` suite.

### Completion Criteria
- [ ] The `git status --short` gate was RE-RUN at this task's own execution
      time (not assumed from TASK-17).
- [ ] If dirty: every deterministic-tier exact-match assertion for HGB is
      `xfail`-marked with a reason citing this plan and the dirty-file list;
      no other assertion is masked.
- [ ] If clean: full pinned-tolerance parity, mirroring TASK-14's rigor.
- [ ] `pytest crates/mlrs-py/python/tests/` full suite green (with `xfail`s
      reporting `XFAIL`, not `XPASS` or a hard `FAIL`).

### Risks and Guardrails
- Risk: an `XPASS` (unexpectedly-passing xfail) silently masking a REAL
  fixture-freshness improvement that should trigger un-xfailing and full
  tolerance-pinning. Mitigation: `strict=False` reports `XPASS` without
  hard-failing the suite, but the Completion Criteria explicitly calls out
  checking for it, not ignoring it.
- Risk (SPEC risk 2): constructing the deterministic-tier HGB estimator
  with the class default `n_bins=64` instead of the required `255` override
  breaks even a CLEAN-fixture deterministic-tier assertion. Mitigation:
  Implementation Step 2 explicitly constructs with `n_bins=255`, matching
  the Rust test's own override.

---

## TASK-25 — PY-ENS-05 (HGB): gate-test updates

- **Spec:** `PY-ENS-05`
- **Order:** 25 (Wave 13, final task)
- **Depends on:** TASK-24
- **Parallel with:** none

### Objective
`test_params.py`/`test_estimator_checks.py` gain the two HGB entries
(`test_shims.py`'s generic fitted-attr list needs NO new entry — HGB has no
new fitted attributes beyond `predict`/`predict_proba`/`classes_`, already
covered structurally by `ALL_SHIMS`'s auto-derivation from TASK-23's
`__all__` update).

### Specification References
- `SPEC-PY-ENS-05` — scenario 5, HGB half.

### Context and Evidence
- Same as TASK-16, HGB-specific.

### Files
- Modify: `crates/mlrs-py/python/tests/test_params.py`
- Modify: `crates/mlrs-py/python/tests/test_estimator_checks.py`

### TDD Sequence
Mirrors TASK-16's structure minus the `feature_importances_`/`oob_score_`
steps (not applicable to HGB).

### Implementation Steps
1. Add the two `EXPECTED_PARAMS` entries (`max_iter, learning_rate,
   max_depth, n_bins, l2_regularization, min_samples_leaf` + defaults).
2. Append `mlrs.HistGradientBoostingClassifier(max_iter=10),
   mlrs.HistGradientBoostingRegressor(max_iter=10)` to `_estimators()`
   (small `max_iter` for the check-sweep's tiny fixtures).
3. Run the check sweep; triage any new failure with a documented xfail
   (never assumed to pass, per SPEC's explicit instruction).
4. Run the full `pytest` suite — **the entire PY-ENSEMBLE plan is complete
   after this task** (modulo TASK-24's HGB-oracle-tolerance gate, which
   remains partially deferred if the churn has not landed by execution
   time — this is a documented, intentional, locked-decision-compliant
   incompleteness, not a plan defect).

### Completion Criteria
- [ ] Both HGB entries added to `EXPECTED_PARAMS` and `_estimators()`.
- [ ] `pytest crates/mlrs-py/python/tests/` full suite green (or every
      failure is a documented, reasoned xfail).
- [ ] All 9 SPEC IDs and all 8 SPEC §6 acceptance scenarios are covered by
      at least one task in this plan (see coverage table in the final
      report).

### Risks and Guardrails
- Same check-sweep triage risk as TASK-16.
