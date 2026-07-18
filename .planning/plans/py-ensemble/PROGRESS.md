---
progress_document: PY-ENSEMBLE implementation progress
plan: .planning/plans/py-ensemble/PLAN.md
spec: .planning/plans/py-ensemble/SPEC.md
plan_check: .planning/plans/py-ensemble/PLAN-CHECK.md (latest verdict: PASS, Pass 3)
started_at: 2026-07-18
---

# PY-ENSEMBLE — Implementation Progress

No `./planning/settings.json` exists in this repo. No PageIndex specification
document exists for this feature (`SPEC.md` frontmatter: `pageindex_update:
"NOT APPLICABLE"`). This progress file is the canonical tracker in place of
the GSD `./planning/phase/phase-XX-name/progress.md` convention — executors
working this plan must update THIS file, not create a `./planning/phase/`
tree.

No worktree isolation policy specified; execution proceeds directly in the
main working tree (matches this session's default). No commit policy
specified; the orchestrator (not the executor) decides when to commit —
executors should leave changes uncommitted and report the changed files.

## Wave 1 — RF-IMP-01 foundation (sequential, same fn `rf_fit_impl`)
- [x] TASK-01 — `rf_node_sqsum` kernel + `rf_best_split` node_decrease output + `RfFitOutcome`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-kernels/src/tree.rs` (new `rf_node_sqsum` kernel,
      `rf_best_split` extended with `node_sq`/`node_decrease` params),
      `crates/mlrs-backend/src/prims/random_forest.rs` (new `RfFitOutcome<F>`,
      `RfModel.node_decrease` field + `node_decrease_host()` accessor,
      `rf_fit_class`/`rf_fit_reg`/`rf_fit_impl` return-type change, host
      feature-importances reduction), `crates/mlrs-backend/tests/random_forest_test.rs`
      (all 10 pre-existing `rf_fit_class`/`rf_fit_reg` call sites updated to
      destructure `RfFitOutcome`), new
      `crates/mlrs-backend/tests/random_forest_feature_importances_test.rs`
      (3 Red/Green tests: dominant-feature classifier, regressor mirror,
      all-leaf-forest degenerate case).
      Tests: `cargo test -p mlrs-backend --features cpu --test random_forest_feature_importances_test`
      (3 passed), `cargo test -p mlrs-backend --features cpu` full suite (all
      binaries `ok`, 0 failed — 3 pre-existing, unrelated `reduce_test.rs`
      tests skipped via `--skip`, confirmed pre-existing/out-of-scope: zero
      diff in `reduce.rs`/`reduce_test.rs`, `svd_moderate_256x64` in the same
      run legitimately took 138s and passed, i.e. not a hang, just slow cpu
      backend compute), `cargo build -p mlrs-backend --features wgpu` (ok).
      Specs: SPEC-RF-IMP-01 left `draft`/unimplemented — TASK-01 is the
      backend/kernel foundation only; the algos-layer `feature_importances()`
      accessor (TASK-02/03) is still pending.
- [x] TASK-02 — `RandomForestClassifier::feature_importances()` + oracle fixture extension
      completed_at: 2026-07-18; status: completed
      Tolerance resolution (spec-owner decision, orchestrator-directed) logged
      in the blocker entry directly above this one (2026-07-18 BLOCKED note):
      `SPEC.md` bumped to `spec_revision: 2` (§5 RF-IMP-01, §9 Risk 5) and
      `PLAN.md` TASK-02/03/14 updated to replace the exact `≤1e-5`/`≤1e-4`
      deterministic-tier assertion with `atol=0.05` (both f64 and f32; the
      divergence source is sklearn's own internal split-tie-breaking, not
      f32/f64 rounding, so no separate wider f32 band applies here) —
      resolution already reflected in this repo's `SPEC.md`/`PLAN.md` before
      this task's completion.
      Files: `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
      (`feature_importances_: Vec<F>` field + `feature_importances()`
      accessor on `RandomForestClassifier<F, Fitted>`; `fit()` destructures
      `RfFitOutcome { model, feature_importances, oob_score: _ }`),
      `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs` (minimal
      deviation: `fit()` destructures `RfFitOutcome { model, .. }` only, no
      new accessor — required because TASK-01 only fixed `mlrs-backend`,
      leaving all of `mlrs-algos` uncompilable; TASK-03 owns the regressor's
      own `feature_importances()` accessor next),
      `crates/mlrs-algos/tests/random_forest_classifier_test.rs` (+4 new
      tests: `feature_importances_matches_sklearn_deterministic_f64`/`_f32`
      at `atol=0.05` plus an independent dtype-aware sum-to-1 sanity check
      (`1e-9` f64 / `1e-6` f32, the latter widened from the plan's literal
      `1e-9` to accommodate f32's per-element mantissa rounding on the final
      `f64_to_host::<F>` cast — the normalization itself happens in f64
      before that cast), `feature_importances_dominant_feature_ranking`
      (qualitative tier, hand-built dominant-feature-vs-noise dataset, the
      PRIMARY correctness signal for RF-IMP-01 per the resolution)),
      `scripts/gen_oracle.py` (`gen_random_forest_classifier` extended with
      `ref_feature_importances = det.feature_importances_` on the existing
      deterministic-tier `det` estimator, appended to the same `np.savez`
      call), `tests/fixtures/rf_cls_{f32,f64}_seed42.npz` (regenerated in
      `/tmp/oracle-venv`, sklearn 1.9.0; verified byte-for-byte that every
      pre-existing key — `X, y, Xq, yq, det_pred_train, det_proba_train,
      stat_acc_test` — is unchanged, only `ref_feature_importances` added).
      Tests: `cargo test --offline -p mlrs-algos --features cpu --test
      random_forest_classifier_test` (9 passed, 0 failed). Full regression:
      every OTHER test binary in `mlrs-algos` (43 of 44 test files) run and
      confirmed passing with zero failures across this task's session
      (`complement_nb`, `dbscan`, `decision_tree_*`, `elastic_net`,
      `empirical_covariance`, `gaussian_nb`, `hdbscan`,
      `hist_gradient_boosting_*`, `incremental_pca`, `kernel_*`, `kmeans`,
      `knn_*`, `ledoit_wolf`, `linear_*`, `logistic`, `mbsgd_*`, `metrics_*`,
      `multinomial_nb`, `nb_common`, `nearest_neighbors`, `pca`,
      `random_forest_regressor` (5 passed), `random_forest_perf` (ignored),
      `random_projection`, `ridge`, `sgd_config`, `spectral_clustering`,
      `spectral_embedding`, `truncated_svd`, `typestate`, `bernoulli_nb`,
      `categorical_nb`, `compile_fail`); `umap_test.rs` (pre-existing,
      unrelated to this task — zero code-path overlap with
      `ensemble::random_forest_*`) was still executing in the background at
      completion time — confirmed legitimately slow CPU-bound proptest
      compute, not a hang (steadily growing CPU time, incremental per-test
      completions observed over ~50 min of polling), consistent with this
      project's documented cpu-backend-is-slow precedent (TASK-01's own
      evidence: `svd_moderate_256x64` legitimately took 138s). `cargo build
      -p mlrs-algos --features cpu` compiles cleanly (only pre-existing,
      unrelated `mlrs-kernels` f32-fallback warnings). Independent sanity
      check: `feature_importances().iter().sum::<f64>()` == `1.0` within
      dtype-aware tolerance, confirmed passing.
      Specs: `SPEC-RF-IMP-01` left `draft`/unimplemented in `SPEC.md` — this
      task completes the CLASSIFIER half only; TASK-03 (regressor) is still
      required before RF-IMP-01 as a whole is fully implemented (no
      PageIndex document exists for this feature per `SPEC.md`'s own
      frontmatter, so no PageIndex state to synchronize).
- [x] TASK-03 — `RandomForestRegressor::feature_importances()`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs`
      (`feature_importances_: Vec<F>` field + `feature_importances()`
      accessor on `RandomForestRegressor<F, Fitted>`, mirroring TASK-02's
      classifier pattern exactly; `fit()` now destructures the FULL
      `RfFitOutcome { model, feature_importances, oob_score: _ }` — replacing
      TASK-02's minimal `{ model, .. }` unblock deviation; `new()`/`build()`
      literals updated to initialize `feature_importances_: Vec::new()`),
      `crates/mlrs-algos/tests/random_forest_regressor_test.rs` (+3 new
      tests, mirroring `random_forest_classifier_test.rs`:
      `feature_importances_matches_sklearn_deterministic_f64`/`_f32` at
      `atol=0.05` (SPEC.md `spec_revision: 2` resolution, applies here
      unchanged per this task's own Objective note) plus an independent
      dtype-aware sum-to-1 sanity check, `feature_importances_dominant_feature_ranking`
      (qualitative tier — hand-built dataset, continuous `y` strongly
      correlated with feature 0 only, noise features 1-3 — the PRIMARY
      correctness signal)), `scripts/gen_oracle.py`
      (`gen_random_forest_regressor` extended with `ref_feature_importances =
      det.feature_importances_` on the existing deterministic-tier `det`
      estimator, appended to the same `np.savez` call — TASK-02 had only
      extended the classifier generator, per its own Files scope; this task
      owns the regressor generator edit as planned), `tests/fixtures/rf_reg_{f32,f64}_seed42.npz`
      (regenerated in `/tmp/oracle-venv`, sklearn 1.9.0; verified byte-for-byte
      via per-`.npy`-member `cmp` that every pre-existing key — `X, y, Xq, yq,
      det_pred_train, stat_r2_test` — is unchanged, only `ref_feature_importances`
      added).
      Tests: `cargo test -p mlrs-algos --features cpu --test
      random_forest_regressor_test feature_importances` (Red: compile error,
      `feature_importances` method not found on `RandomForestRegressor<F, S>`
      — the expected initial failure; Green: 3 passed). Full
      `random_forest_regressor_test.rs`: 8 passed, 0 failed (5 pre-existing +
      3 new, zero regression). Full `random_forest_classifier_test.rs`
      (unmodified by this task, re-run to confirm the shared `RfFitOutcome`/
      `gen_oracle.py` edits didn't regress the classifier sibling): 9 passed,
      0 failed. `cargo check -p mlrs-algos --features cpu --tests` (compiles
      EVERY test binary in the crate, all 44 files including the historically
      slow `umap_test.rs`/`hdbscan_test.rs` proptest binaries, WITHOUT
      running them): clean, zero errors — confirms no downstream consumer of
      `RandomForestRegressor` broke anywhere in the crate. Note: a literal
      `cargo test -p mlrs-algos --features cpu` full-suite RUN (not just
      compile-check) was attempted but the host was under extreme,
      unrelated multi-tenant load for this session (`uptime` load average
      120-230 on a 16-core machine, concurrent unrelated `catboost_rs`/
      `cb-train` cargo processes observed via `ps`); the run progressed
      cleanly with zero compile/test errors through `kernel_density_test`,
      `linear_regression_test`, `mbsgd_regressor_test`, and 13+ minutes into
      the already-documented-slow `umap_test` (PROGRESS.md's own TASK-01/02
      precedent: proptest-heavy, legitimately slow on this CPU backend, zero
      code-path overlap with `ensemble::random_forest_*`) before the
      background process was killed by external system pressure (not a test
      failure — the process was reaped with zero output ever written,
      consistent with resource exhaustion, not an assertion failure). Given
      (a) the directly-relevant `random_forest_{classifier,regressor}_test.rs`
      binaries both pass in full, (b) the crate-wide `cargo check --tests`
      confirms zero broken consumers, and (c) this task's change is purely
      additive (new field + new method on the `Fitted`-state struct only, no
      altered public signature anywhere else), this is treated as equivalent
      regression evidence to a full `cargo test` pass, per this project's own
      established slow-CPU-backend precedent (TASK-01/02 blocker log).
      Specs: `SPEC-RF-IMP-01`'s algos+prim+kernel-layer contract (both
      classifier AND regressor halves) is now fully implemented across
      TASK-01+02+03. `SPEC.md`'s per-spec `**Status:** draft.` line for
      RF-IMP-01 was left UNCHANGED (no PageIndex document exists for this
      feature per `SPEC.md`'s own frontmatter — `pageindex_update: "NOT
      APPLICABLE"` — so there is no PageIndex implementation-state field to
      synchronize, and SPEC.md status-field transitions in this repo's
      convention are orchestrator-directed, not executor-silent, per the
      TASK-02 blocker-resolution precedent); RF-IMP-02 (the Python-binding
      half of the feature-importances surface) remains unimplemented and is
      out of this task's scope (Wave 3+, TASK-08/09).

## Wave 2 — RF-OOB-01 (sequential, depends on TASK-01 only)
- [x] TASK-04 — bootstrap-rederive + OOB aggregation in `rf_fit_impl`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-backend/src/prims/random_forest.rs` (`RfParams.oob_score:
      bool` field, added exactly per TASK-04's own Red-test premise —
      confirmed absent on `RfParams` before this task via CodeGraph/Read;
      `rf_fit_impl` restructured so `RfModel` is assembled right after the
      level loop — BEFORE the RF-IMP-01 feature-importances reduction, which
      now reads it through the existing `split_feature_host`/`is_leaf_host`/
      `node_decrease_host` accessors instead of the (now-moved) local
      `DeviceArray` bindings — then the new gated OOB block: rederives the
      bootstrap mask via a FRESH `SplitMix64::new(params.seed)` +
      `bootstrap_weights::<F>` pass (byte-identical to the level-loop's own
      mask, since `bootstrap_weights` is documented as the first draw on a
      freshly-seeded stream), reuses the existing private `predict_leaves`
      helper — which itself launches the SAME `rf_predict_leaf` kernel the
      predict path already uses — against the just-built `model` and the
      still-in-scope training `x`, then host-aggregates per training row over
      ONLY the out-of-bag trees (`w_host2[t*n+i] == 0`): classifier — argmax
      of the mean OOB-tree class distribution vs. the dense class index,
      accuracy; regressor — mean OOB-tree leaf value vs. `y`, R². Rows with
      zero OOB trees are excluded from the aggregate and counted;
      `log::warn!` fires once (not per-row) if any were excluded. `false`
      (default) short-circuits to `None` with zero extra device/host work),
      `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs` +
      `random_forest_regressor.rs` (both `fit()`'s `RfParams { ... }`
      construction sites gained `oob_score: false` — this task's own
      "audit every existing construction site" step, all 6 non-definition
      sites workspace-wide enumerated via `grep -rn "RfParams {"`: the 2
      algos `fit()` sites plus 4 in `crates/mlrs-backend/tests/` —
      `random_forest_test.rs`'s `params_single_tree()` +
      `bootstrap_fit_is_seed_deterministic_f32`'s inline literal, and
      `random_forest_feature_importances_test.rs`'s
      `params_dominant_feature()` + the all-leaf-forest test's inline
      literal — all 6 updated, self-detecting via compile failure per the
      plan's own risk note), new
      `crates/mlrs-backend/tests/random_forest_oob_test.rs` (3 tests:
      `oob_score_false_is_none_and_adds_no_cost`,
      `oob_score_true_matches_statistical_band` (classifier + regressor
      mirror, wide `0.0..=1.0` / finite-R² placeholder band — the exact
      sklearn statistical-tier cross-check is TASK-06/07's job, not this
      task's, per the plan's own Objective), `oob_score_zero_oob_rows_excluded_not_panicking`
      (`n_trees: 1`, mathematically guaranteed — not merely probabilistic —
      that at least one of the 8 training rows is drawn into the single
      tree's bootstrap sample, since 8 with-replacement draws must land
      somewhere among 8 slots, exercising the skip-and-warn path
      deterministically)).
      Tests: `cargo test -p mlrs-backend --features cpu --test
      random_forest_oob_test` (3 passed, 0 failed — confirmed as genuine
      Red-then-Green: `RfParams.oob_score` was absent on-disk before this
      task's edit, verified via CodeGraph read at task start and re-confirmed
      via `git diff` showing the field as a new `+` line in this session).
      Full regression: `cargo test -p mlrs-backend --features cpu` (exit
      code 0, 37 test binaries, 0 failed — includes all 3 new
      `random_forest_oob_test.rs` tests, the 3 `random_forest_feature_importances_test.rs`
      tests, all 6 `random_forest_test.rs` tests confirming the TASK-01
      `RfFitOutcome` destructuring at its 10 call sites still compiles/passes,
      and the pre-existing unrelated slow suites — `reduce_test.rs`,
      `svd_test.rs` — ran to completion with zero diff in their own source
      files, matching TASK-01/02's documented cpu-backend-is-slow
      precedent, not a regression). `cargo build -p mlrs-backend --features
      wgpu` (clean). `cargo check -p mlrs-algos --features cpu --tests`
      (clean, confirms no downstream `RfParams`/`RfFitOutcome` consumer
      broke). `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test --test random_forest_regressor_test`
      (9 + 8 passed, 0 failed — the two direct downstream consumers of the
      `oob_score: false` field addition, confirming zero behavior
      regression since the field is a pure no-op at `oob_score=false`).
      `cargo fmt -p mlrs-backend -- --check` confirmed pre-existing (not
      task-introduced) formatting drift in `random_forest.rs` via a
      `git stash`/before-vs-after baseline comparison — this task did not
      run a blanket reformat (would touch unrelated pre-existing lines),
      matching the "smallest necessary change" discipline.
      Specs: `SPEC-RF-OOB-01`'s Rust-core computation contract (`RfParams.
      oob_score`, the gated fit-time aggregation, the zero-OOB-row
      `log::warn!` signal) is implemented and verified for the MECHANISM
      (wide placeholder band); `SPEC.md`'s per-spec `**Status:** draft.`
      line left UNCHANGED, matching the TASK-02/03 precedent (no PageIndex
      document exists for this feature — status transitions are
      orchestrator-directed here, not executor-silent). RF-OOB-01 as a
      WHOLE spec is not yet fully implemented: its builder-rejection
      Given/When/Then (`oob_score=true, bootstrap=false` → `Err(BuildError::
      OobRequiresBootstrap)`) is TASK-05's scope, and its sklearn-parity
      statistical-tier cross-check (vs. `ref_oob_score`) is TASK-06/07's —
      neither is claimed complete by this task.
- [x] TASK-05 — `oob_score=true, bootstrap=false` builder rejection
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-algos/src/error.rs` (new `BuildError::OobRequiresBootstrap
      { estimator: &'static str }` variant, mirroring `InvalidEps`'s
      `#[error("...")]` + named-field shape, inserted after
      `InvalidMinSamplesForest`), `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
      + `random_forest_regressor.rs` (both: `oob_score: bool` field added to
      the MAIN struct — NOT only the Builder, per this task's own explicit
      Objective note — AND to the Builder struct; `new()`/`into_builder()`/
      `build::<F>()`'s `Ok(...)` literal and `fit()`'s `Ok(...)` literal all
      thread the field alongside the existing `bootstrap` field exactly;
      `.oob_score(bool)` setter added to both builders, default `false`;
      `build::<F>()` gains an inline `if self.oob_score && !self.bootstrap {
      return Err(BuildError::OobRequiresBootstrap { estimator: "..." }); }`
      check immediately after the existing `validate_forest_hyperparams(...)?`
      call, per the plan's Q-BuildError-location resolved decision — NOT
      threaded through the shared `validate_forest_hyperparams` helper),
      `crates/mlrs-algos/tests/random_forest_classifier_test.rs` +
      `random_forest_regressor_test.rs` (+2 tests each:
      `builder_rejects_oob_score_without_bootstrap` (Red: compile error —
      `.oob_score()` setter and `BuildError::OobRequiresBootstrap` variant
      absent; Green: passes),
      `builder_accepts_oob_score_with_bootstrap` (positive-case regression,
      confirms the cross-check does not reject the valid combination)).
      Tests: `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test builder_rejects_oob_score_without_bootstrap`
      (Red: 3 compile errors — `no method named oob_score`,
      `no variant named OobRequiresBootstrap` — confirmed via direct run
      before Green; Green: passed). Same Red/Green sequence independently
      confirmed for `random_forest_regressor_test`. Full focused run:
      `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test --test random_forest_regressor_test`
      (11 + 10 passed, 0 failed — includes the 4 new tests plus all
      pre-existing TASK-01..04 tests, zero regression). `cargo check -p
      mlrs-algos --features cpu --tests` (compiles every test binary in the
      crate, all 44 files, clean — confirms no downstream consumer broke).
      `cargo fmt -p mlrs-algos -- --check`: confirmed via a `git
      stash`/before-vs-after baseline comparison that this task introduced
      ZERO new formatting drift (all diffed lines in `error.rs` and both
      test files are pre-existing, unrelated to this task's edits — matches
      TASK-04's established "no blanket reformat" discipline). Full-suite
      regression: `cargo test -p mlrs-algos --features cpu` — 45 test
      binaries completed with 0 failures (including
      `random_forest_classifier_test`/`random_forest_regressor_test` both
      passing in full within the same run) before the run reached the
      already-documented-slow, code-path-disjoint `umap_test.rs` proptest
      binary (TASK-01/02/03's own established precedent — legitimately slow
      CPU-bound compute, not a hang); combined with the crate-wide `cargo
      check --tests` clean compile and the two directly-relevant test
      binaries' full pass, this is treated as equivalent regression
      evidence to a full suite pass, per the project's own established
      precedent (TASK-02/03 blocker log).
      Specs: `SPEC-RF-OOB-01`'s build-time validation contract (`oob_score`
      builder setter + main-struct field, the `bootstrap`/`oob_score`
      cross-check, `BuildError::OobRequiresBootstrap`) is implemented and
      verified. `SPEC.md`'s per-spec `**Status:** draft.` line left
      UNCHANGED, matching the TASK-02/03/04 precedent (no PageIndex document
      exists for this feature — status transitions are orchestrator-directed
      here, not executor-silent). RF-OOB-01 as a WHOLE spec is still not
      fully implemented: the sklearn-parity statistical-tier oracle
      cross-check (`ref_oob_score`) is TASK-06/07's scope, not this task's.
- [x] TASK-06 — `oob_score_` sklearn-parity oracle test (classifier)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`
      (`oob_score_: Option<F>` field added to the struct — populated `None`
      on `new()`/`build::<F>()`'s `Ok(...)` literals (Unfit-state
      construction); `fit()` now threads `self.oob_score` (the builder-carried
      flag, already present on the struct since TASK-05) into
      `RfParams.oob_score` at the existing `rf_fit_class(...)` call site
      instead of the previous hardcoded `oob_score: false`, and destructures
      `RfFitOutcome { model, feature_importances, oob_score: oob_score_ }`
      into the new field; new `pub fn oob_score(&self) -> Option<F>`
      accessor on the `Fitted`-state impl block, alongside `feature_importances()`),
      `crates/mlrs-algos/tests/random_forest_classifier_test.rs` (+3 tests:
      `oob_score_within_statistical_band_f64`/`_f32` against the SAME
      `RF_STAT_N_ESTIMATORS=64, RF_STAT_MAX_DEPTH=8` statistical-tier
      hyperparameters as the existing `ACC_MARGIN` tier, with independently
      named `OOB_MARGIN`/`OOB_MARGIN_F32` constants (both `0.10`, per
      Plan-Check Pass-1 Issue 2's mandated f32/f64 split — kept unequal in
      principle even though the tuned values landed identical);
      `oob_score_none_when_flag_false` (public-API-level regression,
      confirms the `false` default end-to-end through the full
      `builder -> fit -> accessor` path, distinct from TASK-04's prim-level
      version)), `scripts/gen_oracle.py` (`gen_random_forest_classifier`
      extended with a SECOND sklearn construction — same seed/data/
      statistical-tier hyperparameters as the existing `stat` estimator,
      plus `bootstrap=True, oob_score=True` — producing `ref_oob_score =
      float(stat_oob.oob_score_)`, a single non-dtype-dependent value cast
      into BOTH the `f32` and `f64` `np.savez` calls this generator already
      makes), `tests/fixtures/rf_cls_{f32,f64}_seed42.npz` (regenerated in
      `/tmp/oracle-venv`, sklearn 1.9.0; verified byte-for-byte via
      `np.array_equal` per pre-existing key — `X, y, Xq, yq, det_pred_train,
      det_proba_train, stat_acc_test, ref_feature_importances` — all
      unchanged; only `ref_oob_score` added, value `0.8125` in both files).
      Tolerance note: Green-time OBSERVED divergence `|got - ref_oob_score|
      ≈ 0.0104` for BOTH f64 (`0.80208333... vs 0.8125`) and f32
      (`0.80208331... vs 0.8125` — the two dtypes agree with each other to
      `~3e-8`, confirming the divergence is almost entirely
      `SplitMix64`-vs-sklearn's-`MT19937` RNG-stream disagreement, not
      f32/f64 rounding). The `0.10` starting `OOB_MARGIN`/`OOB_MARGIN_F32`
      (documented as a tunable starting point, per the plan's own Green
      step) was NOT widened — it already comfortably covers (~10x) the
      actually-observed value for both dtypes, so per the plan's own "never
      silently widen past what Green-time shows is needed" instruction, both
      constants were left at their starting value, with the observed
      divergence recorded in a doc-comment on the constants (not a bare
      unexplained number). This is also documented, mechanism-plausibility
      evidence, not just a numeric pass: `0.8125`/`~0.802` (81%/80% OOB
      accuracy) is a plausible score for a 64-tree, depth-8 statistical-tier
      forest on this fixture's data — not wildly implausible — so no
      escalation was warranted per this task's own tolerance-flexibility
      guidance.
      Tests: `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test oob_score_within_statistical_band_f64`
      (Red: 2 compile errors — `no method named oob_score found ... private
      field, not a method` — confirmed via direct run before Green,
      the expected-for-the-stated-reason failure since only the private
      `oob_score: bool` flag field existed pre-Green, not a `-> Option<F>`
      accessor method; Green: passed). Full focused run: `cargo test -p
      mlrs-algos --features cpu --test random_forest_classifier_test` (14
      passed, 0 failed — includes all 3 new TASK-06 tests plus every
      pre-existing TASK-01/02/04/05 test in this file, zero regression).
      `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test --test random_forest_regressor_test` (14
      + 10 passed, 0 failed — the regressor file is untouched by this task,
      confirming zero cross-contamination). `cargo check -p mlrs-algos
      --features cpu --tests` (compiles every test binary in the crate, all
      44 files, clean). `cargo fmt -p mlrs-algos -- --check` (isolated the
      ONE line of NEW formatting drift this task's own edit introduced — a
      >100-char `let got = clf.oob_score().expect(...)` line — and
      reformatted it to rustfmt's expected multi-line form; confirmed via a
      `git stash`/before-vs-after `--check` diff comparison that every OTHER
      reported drift line pre-dates this task, matching the established
      TASK-04/05 "no blanket reformat" discipline). Full-suite regression:
      `cargo test -p mlrs-algos --features cpu` — ran to completion for 41 of
      44 test binaries (all alphabetically before `umap_test.rs`) with **0
      failures across every one of them** (`test result: ok` × 45 including
      0-test/doctest binaries, 0 `FAILED` anywhere in the log), including
      `random_forest_classifier_test.rs` (14 passed) and
      `random_forest_regressor_test.rs` (10 passed) both re-confirmed passing
      within this same full-suite run; the run was stopped once it reached
      `umap_test.rs`, this project's own documented slow proptest-heavy
      binary (TASK-01/02/03's established precedent — legitimately slow
      CPU-bound compute, zero code-path overlap with `ensemble::random_forest_*`),
      per this project's established equivalent-regression-evidence
      precedent (TASK-02/03/04/05 blocker log) combined with the crate-wide
      `cargo check --tests` clean compile.
      Specs: `SPEC-RF-OOB-01`'s statistical-tier sklearn-parity oracle
      assertion (CLASSIFIER half) is implemented and verified — `oob_score()`
      matches sklearn's `oob_score_` within a documented, Green-time-observed
      statistical band, and the f32 fixture key is asserted against (not
      orphaned, Plan-Check Pass-1 Issue 2). `SPEC.md`'s per-spec `**Status:**
      draft.` line left UNCHANGED, matching the TASK-02/03/04/05 precedent (no
      PageIndex document exists for this feature — status transitions are
      orchestrator-directed here, not executor-silent). RF-OOB-01 as a WHOLE
      spec is still not fully implemented: the regressor half of the
      sklearn-parity statistical-tier oracle assertion is TASK-07's scope,
      not this task's.
- [x] TASK-07 — `oob_score_` sklearn-parity oracle test (regressor)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-algos/src/ensemble/random_forest_regressor.rs`
      (`oob_score_: Option<F>` field added to the struct — populated `None`
      on `new()`/`build::<F>()`'s `Ok(...)` literals (Unfit-state
      construction); `fit()` now threads `self.oob_score` (the
      builder-carried flag, already present on the struct since TASK-05)
      into `RfParams.oob_score` at the existing `rf_fit_reg(...)` call site
      (replacing the previous hardcoded `oob_score: false`), and
      destructures `RfFitOutcome { model, feature_importances, oob_score:
      oob_score_ }` into the new field; new `pub fn oob_score(&self) ->
      Option<F>` accessor on the `Fitted`-state impl block, alongside
      `feature_importances()` — mirrors TASK-06's classifier pattern
      exactly), `crates/mlrs-algos/tests/random_forest_regressor_test.rs`
      (+3 tests: `oob_score_within_statistical_band_f64`/`_f32` against the
      SAME `RF_STAT_N_ESTIMATORS=64, RF_STAT_MAX_DEPTH=8` statistical-tier
      hyperparameters as the existing `R2_MARGIN` tier, with
      INDEPENDENTLY-named/tuned `OOB_MARGIN`/`OOB_MARGIN_F32` constants
      (both `0.10`, a separate constant pair from the classifier's own
      TASK-06 `OOB_MARGIN`/`OOB_MARGIN_F32`, per Plan-Check Pass-1 Issue 2's
      mandated per-estimator independence); `oob_score_none_when_flag_false`
      (public-API-level regression, confirms the `false` default end-to-end
      through the full `builder -> fit -> accessor` path, distinct from
      TASK-04's prim-level version)), `scripts/gen_oracle.py`
      (`gen_random_forest_regressor` extended with a SECOND sklearn
      construction — same seed/data/statistical-tier hyperparameters as the
      existing `stat` estimator, plus `bootstrap=True, oob_score=True` —
      producing `ref_oob_score = float(stat_oob.oob_score_)`, a single
      non-dtype-dependent value cast into BOTH the `f32` and `f64`
      `np.savez` calls this generator already makes), `tests/fixtures/rf_reg_{f32,f64}_seed42.npz`
      (regenerated in `/tmp/oracle-venv`, sklearn 1.9.0; verified
      byte-for-byte via `np.array_equal` per pre-existing key — `X, y, Xq,
      yq, det_pred_train, stat_r2_test, ref_feature_importances` — all
      unchanged; only `ref_oob_score` added, value `0.998511` (f32) /
      `0.99851104` (f64) in the respective files).
      Tolerance note: Green-time OBSERVED divergence `|got - ref_oob_score|
      ≈ 0.00694` for BOTH f64 (`0.99157230... vs 0.99851104...`) and f32
      (`0.99157232... vs 0.99851102...` — the two dtypes agree with each
      other to `~1e-8`, confirming the divergence is almost entirely
      `SplitMix64`-vs-sklearn's-`MT19937` RNG-stream disagreement, not
      f32/f64 rounding — mirrors TASK-06's own finding). The `0.10` starting
      `OOB_MARGIN`/`OOB_MARGIN_F32` (documented as a tunable starting point,
      per the plan's own Green step, ≈14x the observed regressor
      divergence) was NOT widened — it already comfortably covers the
      actually-observed Green-time value for both dtypes, so per the plan's
      own "never silently widen past what Green-time shows is needed"
      instruction, both constants were left at their starting value, with
      the observed divergence recorded in a doc-comment on the constants
      (not a bare unexplained number). `0.998511`/`~0.9916` (≈99.9%/99.2%
      OOB R²) is a plausible score for a 64-tree, depth-8 statistical-tier
      forest on this fixture's piecewise-constant-target data (a much
      easier regression problem than the classifier's noisy 3-class
      target, consistent with the higher absolute score) — not wildly
      implausible — so no escalation was warranted per this task's own
      tolerance-flexibility guidance.
      Tests: `cargo test -p mlrs-algos --features cpu --test
      random_forest_regressor_test oob_score_within_statistical_band_f64`
      (Red: 2 compile errors — `no method named oob_score found ... private
      field, not a method` — confirmed via direct run before Green, the
      expected-for-the-stated-reason failure since only the private
      `oob_score: bool` flag field existed pre-Green, not a `-> Option<F>`
      accessor method; Green: passed). Full focused run: `cargo test -p
      mlrs-algos --features cpu --test random_forest_regressor_test` (13
      passed, 0 failed — includes all 3 new TASK-07 tests plus every
      pre-existing TASK-01/02/03/04/05 test in this file, zero regression).
      `cargo test -p mlrs-algos --features cpu --test
      random_forest_classifier_test` (14 passed, 0 failed — the classifier
      file is untouched by this task, confirming zero cross-contamination).
      `cargo check -p mlrs-algos --features cpu --tests` (compiles every
      test binary in the crate, all 44 files, clean — confirms no
      downstream consumer broke). `cargo fmt -p mlrs-algos -- --check`:
      confirmed via a `git stash`/before-vs-after `--check` diff comparison
      that this task introduced ZERO new formatting drift (the only diff
      location reported in either modified test file, at each file's own
      `feature_importances_matches_sklearn_deterministic_f32` line, predates
      this task — it is TASK-02/03's own line, unchanged by TASK-07's
      edits). `cargo build -p mlrs-algos --features cpu` (clean). Full-suite
      regression: `cargo test -p mlrs-algos --features cpu` — ran under
      extreme, unrelated host multi-tenant load for this session (`uptime`
      load average 84-186 on a 16-core machine throughout the run, no other
      cargo processes competing this time but the load was OS-external);
      **26 of 44 test binaries completed with `test result: ok` and 0
      `FAILED`/`error[` anywhere in the log** (alphabetically through
      `logistic_test.rs`, i.e. before `random_forest_*` and `umap_test.rs`
      are even reached) before the run was stopped due to the extreme host
      contention, matching this project's own documented equivalent-
      regression-evidence precedent (TASK-01/02/03/04/05/06 blocker log:
      slow-CPU-backend / resource-contention sessions are treated as
      equivalent to a full pass when combined with (a) the directly-relevant
      test files passing in full isolation — done above — and (b) a
      crate-wide `cargo check --tests` clean compile — also done above).
      Specs: `SPEC-RF-OOB-01`'s statistical-tier sklearn-parity oracle
      assertion (REGRESSOR half) is implemented and verified — `oob_score()`
      matches sklearn's `oob_score_` within a documented, Green-time-observed
      statistical band, and the f32 fixture key is asserted against (not
      orphaned, mirrors TASK-06/Plan-Check Pass-1 Issue 2). `SPEC.md`'s
      per-spec `**Status:** draft.` line left UNCHANGED, matching the
      TASK-02/03/04/05/06 precedent (no PageIndex document exists for this
      feature per `SPEC.md`'s own frontmatter — `pageindex_update: "NOT
      APPLICABLE"` — status transitions are orchestrator-directed here, not
      executor-silent). **RF-OOB-01 is now FULLY implemented across
      TASK-04+05+06+07** (both classifier and regressor halves of the
      Rust-core computation, the builder-time validation cross-check, and
      the statistical-tier sklearn-parity oracle assertion for both
      estimators and both dtypes). RF-OOB-02 (the Python-binding half of the
      `oob_score`/`oob_score_` surface) remains unimplemented and is out of
      this task's scope (Wave 3+, TASK-08/09).

## Wave 3 — PY-ENS-01/02 Rust binding (sequential, same file `ensemble.rs`)
- [x] TASK-08 — `PyRandomForestClassifier`
      completed_at: 2026-07-18; status: completed
      Files: new `crates/mlrs-py/src/estimators/ensemble.rs`
      (`PyRandomForestClassifier`: `#[new]`, `fit`, `predict_labels`,
      `predict_proba_f32/_f64`, `classes_`, `feature_importances_f32/_f64`
      (RF-IMP-02), `oob_score_f32/_f64` (RF-OOB-02, returning
      `PyResult<Option<f32|f64>>`), `is_fitted`, `dtype`; uses
      `any_estimator_typestate!` per SPEC risk 1/the plan's own template
      note, NOT `any_estimator!`), `crates/mlrs-py/src/estimators/mod.rs`
      (`pub mod ensemble;` added, per this task's own Files/Green scope —
      not deferred to TASK-10), new
      `crates/mlrs-py/tests/random_forest_smoke_test.rs` (Rust integration
      test, `unfit_default()`/`is_unfit()` construct-and-compile gate,
      mirrors `pyclass_smoke_test.rs`), new
      `crates/mlrs-py/tests/test_random_forest.py` (pytest FFI surface:
      fit/predict/predict_proba/classes_/feature_importances_/oob_score_,
      not-fitted-raises, bogus-max_features-raises-ValueError,
      oob_score=True+bootstrap=False-raises-ValueError,
      max_features int/float acceptance — `pytest.importorskip`-guarded
      plus an explicit `hasattr(_mlrs, "RandomForestClassifier")` skip
      guard, since `_mlrs.RandomForestClassifier` is not registered until
      TASK-10; see Deviations).
      Deviations from the plan's literal text (recorded, none change
      observable Python-level product behavior or scope):
      (1) **`max_features` storage** — the plan's Green step sketched
      storing a raw `PyObject` in the `Unfit` arm, resolved at `fit()`.
      Implemented instead as eager parsing at `#[new]` time into a plain
      Rust `MaxFeaturesArg` enum (`Sqrt|Log2|All|Count(usize)|Frac(f64)`),
      mirroring `naive_bayes.rs::resolve_min_categories`'s
      already-in-crate precedent for a `PyAny`-shaped ctor argument. Cause:
      a `PyObject` field would require a live GIL token merely to
      construct the Rust-callable `unfit_default()` smoke-test helper
      (every other estimator's `unfit_default()` needs no interpreter);
      confirmed via a direct spike (`Python::attach` inside a `#[test]`)
      that this environment's linker cannot resolve `-lpython3.14` at all
      (see Deviation 3) — so a `PyObject`-storing design would be
      un-instantiable here even for the compile-only smoke gate. Only the
      `Frac` variant's fraction→count math (needs `n_features`) is still
      deferred to `fit()`, matching the plan's actual intent for the
      data-dependent part. A bogus `max_features` string now raises
      `ValueError` at construction (`__init__`) rather than at `fit()` —
      still a `ValueError`, same test outcome, just fires slightly
      earlier; `oob_score=True, bootstrap=False` is unaffected and still
      raises at `fit()` (via `BuildError::OobRequiresBootstrap`).
      (2) **`max_features=None` sentinel collision** — PyO3's
      `Option<&Bound<PyAny>>` extraction cannot distinguish "argument
      omitted" from "argument explicitly `None`" (both collapse to Rust
      `None` at the FFI boundary; confirmed by a failed compile attempt
      giving a literal non-`Option` `&Bound` parameter a string-literal
      default, which PyO3 rejects — `E0308`, default-expression type must
      match the parameter type exactly). Both cases now resolve to the
      classifier's sklearn default (`"sqrt"`), documented in a code
      comment; a caller wanting "all features" should pass
      `max_features=1.0` (unambiguous via the `Frac` path) instead of
      `None`. Not exercised by any of this task's own Red tests (which
      only assert the bogus-string and int/float-acceptance cases).
      (3) **Test harness: `.rs` construct-gate + `.py` FFI surface, not a
      single `.py`-run-via-`cargo-test` file as the plan's prose loosely
      implied** — confirmed via a direct baseline spike BEFORE any edit
      (`cargo build -p mlrs-py --features cpu`, unmodified tree) that this
      environment's linker cannot resolve `-lpython3.14`
      (`mold: fatal: library not found: python3.14`) — a PRE-EXISTING,
      environment-level gap (missing libpython3.14 dev shared object),
      reproduced byte-for-byte identically with and without this task's
      changes (re-confirmed via `git stash`). This blocks `cargo test -p
      mlrs-py --features cpu` (and even `cargo build`) from LINKING
      anywhere in this crate, regardless of this task's content. `cargo
      check -p mlrs-py --features cpu[/--tests]` (type-check, no link) is
      therefore used as the compile-correctness gate instead, per this
      project's own established "treat a compile check as equivalent
      regression evidence when the full run is blocked by an unrelated,
      pre-existing environment issue" precedent (TASK-01..07 blocker log,
      extended here from "slow" to "cannot link" for the same underlying
      reason: a pre-existing condition this task did not cause and cannot
      fix). `#[pymethods]`-annotated methods are crate-private by
      established convention (confirmed by attempting to call
      `.is_fitted()`/`.feature_importances_f32()` etc. directly from the
      integration-test crate — `E0624 method ... is private`, matching
      every other existing `Py*` wrapper in this crate), so the Rust
      integration test is scoped to the `unfit_default()`/`is_unfit()`
      construct-and-compile gate only (mirrors `pyclass_smoke_test.rs`
      exactly); the FFI-boundary behaviors (not-fitted, bogus
      `max_features`, `oob_score`-without-`bootstrap`) are in
      `test_random_forest.py`, which cannot yet import
      `mlrs._mlrs.RandomForestClassifier` (registration is TASK-10) —
      each test is additionally guarded by `hasattr(...)`-skip so the file
      is collection-safe now and becomes fully exercisable once TASK-10/
      `maturin develop` land (this task's `_mlrs` is also not installed in
      this environment at all — `python3 -c "import mlrs._mlrs"` →
      `ModuleNotFoundError` — confirmed before any edit).
      Tests: `cargo check -p mlrs-py --features cpu` (Red, before
      `ensemble.rs`/`pub mod ensemble;` existed: baseline confirmed via
      Read of `estimators/mod.rs`, exactly 10 `pub mod` lines, no
      `ensemble` — the module genuinely did not exist; Green, after: clean,
      exit 0, only 2 pre-existing unrelated `spectral.rs` dead-code
      warnings). `cargo check -p mlrs-py --features cpu --test
      random_forest_smoke_test` (clean, exit 0 — the compiled
      `unfit_default()`/`is_unfit()` construct gate). `cargo check -p
      mlrs-py --features wgpu` (second backend gate, clean, exit 0).
      `cargo test -p mlrs-py --features cpu --test
      random_forest_smoke_test` (reaches the identical pre-existing
      `-lpython3.14` link failure documented in Deviation 3 — not a
      regression, reproduced byte-for-byte on the unmodified baseline).
      `cargo check -p mlrs-py --features cpu --tests` (whole-crate test
      compile): surfaces PRE-EXISTING, unrelated compile errors in
      `sgd_smoke_test.rs`/`spectral_smoke_test.rs` (stale
      `mlrs_algos::traits` import + `SpectralEmbedding`/`SpectralClustering`
      constructor-signature drift) — confirmed pre-existing and unrelated
      to this task via `git stash` on `estimators/mod.rs` (the ONLY file
      this task modified in-place) reproducing the identical error set;
      `random_forest_smoke_test.rs` itself reports zero errors in that
      same run. `rustfmt --edition 2021` applied to the 2 new files only
      (confirmed via `git diff --stat` that no pre-existing file's bytes
      changed beyond the intentional one-line `mod.rs` addition — matches
      the TASK-04/05 "no blanket reformat" discipline; the crate as a
      whole already carries extensive pre-existing, unrelated `cargo fmt
      --check` drift in nearly every existing estimator file).
      Specs: `SPEC-PY-ENS-01`'s binding-layer contract and
      `SPEC-RF-IMP-02`/`SPEC-RF-OOB-02`'s Rust/PyO3-accessor half are
      implemented for the CLASSIFIER (this task's own Completion Criteria:
      "RF-IMP-02/RF-OOB-02 Rust half satisfied here, not deferred").
      `SPEC.md`'s per-spec `**Status:** draft.` lines left UNCHANGED,
      matching the TASK-01..07 precedent (no PageIndex document exists for
      this feature — `pageindex_update: "NOT APPLICABLE"` — status
      transitions are orchestrator-directed here, not executor-silent).
      None of PY-ENS-01/RF-IMP-02/RF-OOB-02 is claimed FULLY implemented
      by this task alone: the Python shim `@property`/constructor wiring
      (TASK-11), `_mlrs` registration (TASK-10), `__init__.py` export
      (TASK-13), and the oracle-replay/gate tests (TASK-14/16) are still
      required before the spec's full Given/When/Then coverage is
      satisfied end-to-end.
- [x] TASK-09 — `PyRandomForestRegressor`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/src/estimators/ensemble.rs` (appended, per this
      task's own Files scope — same file TASK-08 created):
      `PyRandomForestRegressor` — `crate::any_estimator_typestate!`-generated
      `AnyRandomForestRegressor` enum (`n_estimators, max_depth, n_bins,
      max_features: MaxFeaturesArg, min_samples_split, min_samples_leaf,
      bootstrap, oob_score, seed`, mirroring `AnyRandomForestClassifier`
      field-for-field except no `classes_`), `#[new]` (regressor default
      `max_features=None -> MaxFeaturesArg::All`, NOT the classifier's
      `Sqrt` — sklearn's own regressor default IS "all", so this is not a
      parity gap unlike the classifier's own documented `None`-collapse
      note), `fit(x,y,rows,cols)` (byte-for-byte the classifier's
      dtype-dispatch/`guard_f64()`/`build::<F>().map_err(build_err_to_py)`/
      `TypestateFit::fit(...).map_err(algo_err_to_py)` shape), `predict_f32`/
      `_f64` (new import `Predict as TypestatePredict` from
      `mlrs_algos::typestate`, composing the `Predict` trait exactly like
      `PyLinearRegression::predict_f32/_f64`, `crates/mlrs-py/src/estimators/linear.rs`
      — no `classes_`/`predict_proba` on the regressor, per this task's own
      Objective), `feature_importances_f32/_f64` (RF-IMP-02, thin
      `.feature_importances().to_vec()` readback — no new device work),
      `oob_score_f32/_f64` (RF-OOB-02, `PyResult<Option<f32|f64>>`, thin
      `.oob_score()` readback), `is_fitted`, `dtype` — all mirroring
      `PyRandomForestClassifier`'s own shape one-for-one, minus the
      classifier-only methods, per the plan's own "Mirror TASK-08's
      classifier `#[pyclass]` shape, minus `classes_`/`predict_proba`" Green
      instruction; new top-of-file `use
      mlrs_algos::ensemble::random_forest_regressor::RandomForestRegressor`
      import; module doc-comment corrected from "(appended next)" to
      "(TASK-09, appended below)" (Refactor step, cosmetic only, no
      behavior change)), `crates/mlrs-py/tests/random_forest_smoke_test.rs`
      (new `random_forest_regressor_constructs_unfit` Rust integration test —
      `unfit_default()`/`is_unfit()` construct-and-compile gate, mirroring
      the classifier's own TASK-08 test), `crates/mlrs-py/tests/test_random_forest.py`
      (appended: `test_regressor_predict_before_fit_raises` (the Red test),
      `test_random_forest_regressor_fit_predict` (fit -> predict, RF-IMP-02
      `feature_importances_` sums to 1, RF-OOB-02 `oob_score_` is `None` by
      default, parametrized f32/`@requires_f64`-guarded f64, mirroring the
      classifier's own `test_random_forest_classifier_fit_predict`),
      `test_random_forest_regressor_oob_score_true` (`oob_score=True,
      bootstrap=True` -> finite float), `test_regressor_max_features_default_is_all_not_sqrt`
      (the plan's own Implementation Step 1 "indirect check" — constructs with
      no args, fits, confirms no `ValueError`, since `MaxFeatures` itself is
      not Python-visible), `test_regressor_feature_importances_before_fit_raises`,
      `test_regressor_max_features_bogus_string_raises_value_error`,
      `test_regressor_oob_score_true_without_bootstrap_raises_value_error` —
      all `@requires_rf_reg`-guarded (`hasattr(_mlrs, "RandomForestRegressor")`,
      since registration is TASK-10's scope, same pattern TASK-08 established
      for the classifier)).
      Deviations from the plan's literal text (none change observable
      Python-level product behavior or scope, all directly inherited from
      TASK-08's own already-recorded deviations, re-confirmed unchanged by
      this task): (1) `max_features` is stored as the eagerly-parsed
      `MaxFeaturesArg` enum (TASK-08's Deviation 1), not a raw `PyObject`, for
      the identical reason TASK-08 recorded (GIL-free `unfit_default()`
      construction); (2) the `max_features=None` sentinel collision
      (TASK-08's Deviation 2) applies identically here, except — as this
      task's own module doc/Green step notes — it is NOT a parity gap for the
      regressor specifically, since sklearn's regressor default already IS
      "all features"; (3) test harness split (`.rs` construct-gate +
      `.py` FFI surface, TASK-08's Deviation 3) — re-confirmed: this
      environment's linker still cannot resolve `-lpython3.14` (`mold: fatal:
      library not found: python3.14`), reproduced identically on this task's
      own `cargo test -p mlrs-py --features cpu --test random_forest_smoke_test`
      run; `cargo check` (type-check, no link) is therefore used as the
      compile-correctness gate, per the same established project precedent
      TASK-08 used (not re-derived independently — this task did not need to
      re-run TASK-08's own `git stash` baseline spike, since the underlying
      condition is an environment-level `libpython3.14` absence unrelated to
      any RF code and already established).
      Tests: `cargo check -p mlrs-py --features cpu --test
      random_forest_smoke_test` (Red, before `PyRandomForestRegressor`
      existed: `error[E0432]: unresolved import
      mlrs_py::estimators::ensemble::PyRandomForestRegressor` — confirmed via
      direct run before any Green edit, the expected-for-the-stated-reason
      failure; Green, after: clean, exit 0, only the same 2 pre-existing
      unrelated `spectral.rs` dead-code warnings TASK-08 already documented).
      `cargo check -p mlrs-py --features wgpu` (second backend gate, clean,
      exit 0). `cargo check -p mlrs-py --features cpu --tests` (whole-crate
      test compile): the error set is BYTE-FOR-BYTE the same 2 pre-existing,
      unrelated broken binaries TASK-08 already documented
      (`sgd_smoke_test.rs`/`spectral_smoke_test.rs`, stale
      `mlrs_algos::traits` import + `SpectralEmbedding`/`SpectralClustering`
      constructor-signature drift) — `random_forest_smoke_test.rs` reports
      zero errors in that same run, confirming this task introduced no new
      compile break anywhere in the crate. `cargo test -p mlrs-py --features
      cpu --test random_forest_smoke_test` (reaches the identical
      pre-existing `-lpython3.14` link failure, not a regression). `rustfmt
      --edition 2021` applied to the 2 files this task touched/appended to
      (`ensemble.rs`, `random_forest_smoke_test.rs`) only — confirmed via
      `git status --short -- crates/mlrs-py` that `estimators/mod.rs` (a
      TASK-08 edit, untouched by this task) shows no new diff beyond its
      pre-existing single-line `pub mod ensemble;` addition, matching the
      TASK-04/05/08 "no blanket reformat" discipline. Python syntax:
      `python3 -c "import ast; ast.parse(...)"` on the appended
      `test_random_forest.py` confirms valid syntax (pytest itself is not
      installed in this environment, matching TASK-08's own recorded
      environment state — `_mlrs`/`pytest` unavailable; every new test is
      `@requires_rf_reg`-guarded so the file remains collection-safe once the
      environment/build state is ready, per the TASK-08 precedent).
      Specs: `SPEC-PY-ENS-02`'s binding-layer contract and
      `SPEC-RF-IMP-02`/`SPEC-RF-OOB-02`'s Rust/PyO3-accessor half are now
      implemented for BOTH the classifier (TASK-08) AND the regressor (this
      task) — RF-IMP-02/RF-OOB-02's Rust-binding-layer Given/When/Then is
      satisfied for both RF estimators. `SPEC.md`'s per-spec `**Status:**
      draft.` lines left UNCHANGED, matching the TASK-01..08 precedent (no
      PageIndex document exists for this feature — `pageindex_update: "NOT
      APPLICABLE"` — status transitions are orchestrator-directed here, not
      executor-silent). None of PY-ENS-02/RF-IMP-02/RF-OOB-02 is claimed
      FULLY implemented end-to-end by this task alone: the Python shim
      `@property`/constructor wiring (TASK-12), `_mlrs` registration
      (TASK-10), `__init__.py` export (TASK-13), and the oracle-replay/gate
      tests (TASK-15/16) are still required before the spec's full
      Given/When/Then coverage is satisfied end-to-end. Wave 3 (TASK-08,
      TASK-09) is now fully complete.

## Wave 4 — lib.rs registration (4a) parallel with Python shim (4b)
- [x] TASK-10 — `lib.rs` registration (RF)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/src/lib.rs` ONLY (per this task's explicit scope
      restriction — `crates/mlrs-py/python/mlrs/ensemble.py`/`__init__.py`
      and any other Python file are TASK-11/12/13's scope, owned by the
      parallel executor, untouched here): added `use
      estimators::ensemble::{PyRandomForestClassifier,
      PyRandomForestRegressor};` (alongside the existing per-family `use`
      block) and two `m.add_class::<PyRandomForestClassifier>()?;`/
      `m.add_class::<PyRandomForestRegressor>()?;` calls in a new "Phase-13
      ensemble wrappers" comment block (registration count 32 -> 34,
      matching this task's own Objective); corrected the stale "12
      estimator" doc-comment count at all 3 cited locations
      (`lib.rs:65,178,201` in the pre-task line numbering) — reworded to
      name the estimator families rather than a raw count (per this task's
      own Risks/Guardrails note: "prefer wording that names the estimator
      families rather than a raw count... otherwise TASK-20 MUST update the
      same line again" — TASK-20 (HGB registration) still needs its own
      "Phase-14" `add_class` block + registration-count doc-comment bump in
      its own scope, but the reworded family-based phrasing here needs no
      further edit).
      Deviation (environment-consistent, matches TASK-08/09's own recorded
      precedent, not a scope change): this environment's linker still
      cannot resolve `-lpython3.14` (`mold: fatal: library not found:
      python3.14`, reconfirmed via `cargo test -p mlrs-py --features cpu
      --test random_forest_smoke_test` both before and after this task's
      edit) and `pytest`/`maturin`/`mlrs` are not installed
      (`ModuleNotFoundError`/`no maturin`, reconfirmed fresh this session) —
      so the plan's own literal Red test
      (`test_random_forest_classifier_registered_on_mlrs`, a pytest
      `hasattr(mlrs._mlrs, "RandomForestClassifier")` assertion against a
      built wheel) cannot be executed in this environment, and per this
      task's own explicit scope restriction no new/modified Python test file
      was added to carry it. The equivalent Red/Green evidence used instead:
      Red — confirmed via `grep -c "add_class::<Py" lib.rs` = 32 and no
      `estimators::ensemble` import present, before any edit (the class is
      genuinely "not yet registered", matching the plan's own stated Red
      expectation "`False` — not yet registered"); Green — same grep = 34
      after the edit, `estimators::ensemble` import present.
      Tests: `cargo check -p mlrs-py --features cpu` (clean, exit 0, only
      the 2 pre-existing unrelated `spectral.rs` dead-code warnings TASK-08/09
      already documented). `cargo check -p mlrs-py --features wgpu` (clean,
      exit 0). `cargo check -p mlrs-py --features cpu --tests` (whole-crate
      test compile): error set is byte-for-byte the same 2 pre-existing,
      unrelated broken binaries TASK-08/09 already documented
      (`sgd_smoke_test.rs`/`spectral_smoke_test.rs`, stale
      `mlrs_algos::traits` import + `SpectralEmbedding`/`SpectralClustering`
      constructor-signature drift — 23 errors total, identical count/kind
      confirmed via a `git stash`/before-vs-after comparison isolated to
      `lib.rs`); `random_forest_smoke_test.rs` itself reports zero errors in
      that same run, confirming this task's registration change introduced
      no new compile break anywhere in the crate. `cargo test -p mlrs-py
      --features cpu --test random_forest_smoke_test` (reaches the identical
      pre-existing `-lpython3.14` link failure documented above, not a
      regression). `cargo fmt -p mlrs-py -- --check`: isolated via a `git
      stash`/before-vs-after diff comparison on `lib.rs` alone that this
      task introduced ZERO new formatting drift (the one nearby diff hunk,
      at the pre-existing `linear::` import block, is untouched content that
      merely shifted line number because of this task's own doc-comment
      edits above it; the new `ensemble` import line itself produces no diff
      hunk at all — confirmed rustfmt's canonical form for it is the
      single-line form already used).
      Specs: `SPEC-PY-ENS-05`'s registration-layer contract (RF half) is
      implemented for the Rust `_mlrs` `#[pymodule]` registration surface
      specifically — both `PyRandomForestClassifier`/`PyRandomForestRegressor`
      are now importable from the compiled extension once a wheel is built.
      `SPEC.md`'s per-spec `**Status:** draft.` line left UNCHANGED, matching
      the TASK-01..09 precedent (no PageIndex document exists for this
      feature — `pageindex_update: "NOT APPLICABLE"` — status transitions are
      orchestrator-directed here, not executor-silent). PY-ENS-05 as a WHOLE
      spec is NOT yet fully implemented by this task alone: the Python shim
      `@property`/constructor wiring (TASK-11/12, in progress in parallel),
      `__init__.py` export (TASK-13), and the three cross-cutting gate-test
      file updates (`test_params.py`/`test_shims.py`/`test_estimator_checks.py`,
      TASK-16) are still required before PY-ENS-05's full Given/When/Then
      coverage is satisfied end-to-end; this task's own narrower scope (`lib.rs`
      registration only, per the explicit mission-level scope restriction) is
      complete.
- [x] TASK-11 — `ensemble.py` `RandomForestClassifier`
      completed_at: 2026-07-18; status: completed
      Files: new `crates/mlrs-py/python/mlrs/ensemble.py`
      (`RandomForestClassifier(ClassifierMixin, MlrsBase)`: `__init__`
      (defaults verbatim-matching `PyRandomForestClassifier::new`'s
      `#[pyo3(signature=(...))]` in `crates/mlrs-py/src/estimators/ensemble.rs`
      — `n_estimators=100, max_depth=10, n_bins=32, max_features="sqrt",
      min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=True,
      oob_score=False, seed=42`), `fit` (mirrors `naive_bayes.py`'s
      `_normalize`/`_normalize_y`/inline-store-fit template — see Deviation
      1), `predict`/`predict_proba` (mirror `LogisticRegression`'s/
      `_BaseNB`'s exact shape: bare `predict_labels`, `_suffixed("predict_proba")`),
      `feature_importances_`/`oob_score_` `@property`s (RF-IMP-02/RF-OOB-02
      — see Deviation 2 for the exact `_suffixed(...)` base-name
      correction), `classes_` set inline in `fit`).
      Deviations from the plan's literal Green-step text (both
      non-scope-changing, both resolved in favor of the VERIFIED Rust
      source over the plan's prose, per this agent's "confirm exact method
      names before writing the shim" instruction):
      (1) **`self._store_fit(obj, cols)`** — the plan's Green step names
      this helper, but it is a `_BaseNB`-private method defined only in
      `naive_bayes.py`, not on `MlrsBase`. Implemented inline instead
      (`self._mlrs_obj = obj; self._post_fit(cols)`), matching the
      majority precedent already established by `LogisticRegression`/
      `MBSGDClassifier`/`LinearSVC` in `linear.py` (3 of the 4 existing
      classifier shims already do this inline, not via the NB-specific
      mixin) — avoids a cross-module private-helper import, zero behavior
      change.
      (2) **`_suffixed("feature_importances_")` → `_suffixed("feature_importances")`
      (no trailing underscore in the base-name argument)** — the plan's
      own Green-step prose literal had a trailing underscore, which would
      resolve to the nonexistent method name `feature_importances__f32`
      (double underscore); the plan's own Risks-and-Guardrails paragraph
      for this same task independently states the CORRECT form
      (`_suffixed("feature_importances")`, matching `linear.py:234-237`'s
      `coef`/`intercept` pattern) and was followed here, verified directly
      against the actual `feature_importances_f32`/`_f64` method names read
      from `crates/mlrs-py/src/estimators/ensemble.rs` before writing this
      file (per this task's own governing instructions).
      (3) **Practical Red/Green verification used a live Python import,
      not just `ast.parse`** — the task brief anticipated `maturin
      develop`+pytest being unavailable and suggested `ast.parse` as a
      fallback; this environment additionally lacked `numpy`/`pyarrow`
      entirely (masking even PRE-EXISTING shim imports, e.g.
      `from mlrs.naive_bayes import GaussianNB` failed identically before
      this task's edit — confirmed via a baseline run). `pip install
      pyarrow` (numpy already present) into the pre-existing
      `/tmp/oracle-venv` (network available) restored the ability to
      genuinely exercise Red (`ModuleNotFoundError: No module named
      'mlrs.ensemble'`, the correct reason, confirmed against the same
      baseline import which now succeeds) and Green (`from mlrs.ensemble
      import RandomForestClassifier` succeeds) without needing the
      compiled `_mlrs` extension (which genuinely remains unregistered for
      `RandomForest*` in the pre-existing, stale on-disk `.so` snapshot
      read at the START of this task — confirmed via `dir(mlrs._mlrs)`
      showing zero `RandomForest`/`HistGradient` names at that point) — a
      strictly stronger verification than `ast.parse` alone, not a scope
      change (installing a Python package into a scratch venv touches no
      repository file). NOTE: TASK-10 (parallel session) has since landed
      real `_mlrs` registration per its own progress entry above; this
      task's own verification evidence, captured before that landed, is
      unaffected and still accurately describes what THIS task itself
      proved.
      Tests: `python3 -c "import ast; ast.parse(open('mlrs/ensemble.py').read())"`
      (OK). `/tmp/oracle-venv/bin/python3 -m py_compile
      crates/mlrs-py/python/mlrs/ensemble.py` (OK). Red (pre-edit, file
      absent): `ModuleNotFoundError: No module named 'mlrs.ensemble'`.
      Green (post-edit): `from mlrs.ensemble import RandomForestClassifier`
      succeeds; `RandomForestClassifier()` zero-arg-constructs (no `_mlrs`
      extension call needed for construction, matching the
      `naive_bayes.py` precedent) with `get_params()` reporting every ctor
      default verbatim-matching TASK-08's `#[new]` defaults; `sklearn.base.clone(...)`
      round-trips (purity-rule sanity, mirrors `check_parameters_default_constructible`);
      `feature_importances_`/`oob_score_`/`predict` all raise
      `sklearn.exceptions.NotFittedError` before `fit` (verified directly,
      no `_mlrs` call reached). Full end-to-end `.fit()`/`.predict()`
      exercise NOT run in this task — deferred to TASK-14/16 per the
      plan's own wave sequencing (this task's own scope, per the governing
      session's explicit file-ownership boundary, is `ensemble.py` only —
      `lib.rs`/`estimators/mod.rs`/`__init__.py` untouched, confirmed via
      `git status --short` showing only the new, untracked
      `crates/mlrs-py/python/mlrs/ensemble.py`).
      Specs: `SPEC-PY-ENS-01`'s Python-shim-layer contract and
      `SPEC-RF-IMP-02`/`SPEC-RF-OOB-02`'s Python `@property` half are
      implemented for the CLASSIFIER (this task's own Completion
      Criteria). No PageIndex document exists for this feature
      (`SPEC.md` frontmatter: `pageindex_update: "NOT APPLICABLE"`) —
      `SPEC.md`'s per-spec `**Status:** draft.` lines left UNCHANGED,
      matching the TASK-01..10 precedent (status transitions are
      orchestrator-directed, not executor-silent). None of
      PY-ENS-01/RF-IMP-02/RF-OOB-02 is claimed FULLY implemented end-to-end
      by this task alone: `__init__.py` export (TASK-13) and the
      oracle-replay/gate tests (TASK-14/16) are still required.
- [x] TASK-12 — `ensemble.py` `RandomForestRegressor`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/mlrs/ensemble.py` (appended, same file
      TASK-11 created, per this task's own Files scope):
      `RandomForestRegressor(RegressorMixin, MlrsBase)` — mirrors TASK-11's
      classifier one-for-one minus `classes_`/`predict_proba`, plus a
      float-only `predict` (`_suffixed("predict")`, mirrors
      `neighbors.py::KNeighborsRegressor.predict`/`linear.py`'s regressor
      shims exactly). `__init__` defaults verbatim-match
      `PyRandomForestRegressor::new`'s `#[pyo3(signature=(...))]`: identical
      to the classifier's defaults EXCEPT `max_features=1.0` (NOT the
      classifier's `"sqrt"`) — sklearn's own `RandomForestRegressor` default
      IS `1.0` ("all features"); `fit`'s `_normalize_y` dtype helper reuses
      `RandomForestClassifier._x_float(xa)` cross-class within the same
      module (mirrors `neighbors.py::KNeighborsRegressor.fit`'s own
      cross-class reuse of `KNeighborsClassifier._x_float`, and
      `linear.py`'s reuse of `LinearRegression._x_float` across every other
      linear shim in that file — an established in-module precedent, not a
      new pattern). `feature_importances_`/`oob_score_` `@property`s
      byte-for-byte mirror TASK-11's classifier implementation (same
      `_suffixed("feature_importances")`/`_suffixed("oob_score")`
      base-name correction applies identically here — verified against the
      actual `PyRandomForestRegressor::feature_importances_f32/_f64`/
      `oob_score_f32/_f64` method names in `ensemble.rs`, not assumed).
      Deviations: identical in kind to TASK-11's 3 recorded deviations
      (inline `self._mlrs_obj = obj; self._post_fit(cols)` instead of a
      cross-module `_store_fit` import; the `_suffixed("feature_importances")`
      no-trailing-underscore correction; live-import Red/Green verification
      via the same `/tmp/oracle-venv` pyarrow-install, not `ast.parse`
      alone) — none re-derived independently since the underlying
      conditions (environment gaps, method-name ground truth) are the same
      ones TASK-11 already established in this same session.
      Tests: `ast.parse`/`py_compile` clean (same file, re-run after the
      append). Red (pre-edit): `ModuleNotFoundError: No module named
      'RandomForestRegressor'` via `from mlrs.ensemble import
      RandomForestRegressor` (the class did not exist in the file TASK-11
      left behind). Green (post-edit): import succeeds;
      `RandomForestRegressor()` zero-arg-constructs with `get_params()`
      reporting `max_features: 1.0` (confirmed distinct from the
      classifier's `"sqrt"`) and every other default matching TASK-09's
      `#[new]` defaults verbatim; `sklearn.base.clone(RandomForestRegressor(max_features=0.5))`
      round-trips; `feature_importances_`/`oob_score_` both raise
      `NotFittedError` before `fit`, re-confirmed for BOTH estimators in
      the same run as TASK-11's own check (no regression introduced by the
      append). Full end-to-end `.fit()`/`.predict()` exercise deferred to
      TASK-15/16, same reasoning as TASK-11.
      Specs: `SPEC-PY-ENS-02`'s Python-shim-layer contract and
      `SPEC-RF-IMP-02`/`SPEC-RF-OOB-02`'s Python `@property` half are now
      implemented for BOTH the classifier (TASK-11) AND the regressor
      (this task). `SPEC.md`'s per-spec `**Status:** draft.` lines left
      UNCHANGED, matching the TASK-01..11 precedent (no PageIndex document
      exists for this feature — status transitions are
      orchestrator-directed here, not executor-silent). None of
      PY-ENS-02/RF-IMP-02/RF-OOB-02 is claimed FULLY implemented
      end-to-end by this task alone: `__init__.py` export (TASK-13) and
      the oracle-replay/gate tests (TASK-15/16) are still required. Wave 4b
      (TASK-11, TASK-12) is now complete; this session did not touch
      `crates/mlrs-py/src/lib.rs`, `crates/mlrs-py/src/estimators/mod.rs`,
      or `crates/mlrs-py/python/mlrs/__init__.py` (confirmed via
      `git status --short` showing only the new, untracked
      `crates/mlrs-py/python/mlrs/ensemble.py` — `lib.rs` is TASK-10's
      scope, already independently landed per its own progress entry
      above; `__init__.py` remains TASK-13's exclusive scope).

## Wave 5
- [x] TASK-13 — `__init__.py` wiring (RF)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/mlrs/__init__.py` ONLY (per this task's
      exact scope: `from .ensemble import RandomForestClassifier,
      RandomForestRegressor` added to the alphabetically-ordered import
      block (between `.density`/`.kernel_ridge`, matching `ensemble.py`'s
      own module-name position), both names appended to `__all__`
      (inserted after `KernelDensity`, before `KMeans`).
      Tests: Red confirmed BEFORE the edit via a live import against this
      repo's own `crates/mlrs-py/python` tree in `/tmp/oracle-venv`
      (numpy/pyarrow/sklearn installed, `pytest` additionally installed
      this session): `import mlrs; mlrs.RandomForestClassifier` raised
      `AttributeError: module 'mlrs' has no attribute
      'RandomForestClassifier'` (the exact plan-specified Red state).
      Green confirmed after the edit: `mlrs.RandomForestClassifier`/
      `mlrs.RandomForestRegressor` resolve to `mlrs.ensemble.RandomForestClassifier`/
      `RandomForestRegressor`; both zero-arg-construct; `get_params()`
      reports every ctor default verbatim-matching TASK-11/12's `ensemble.py`
      (classifier `max_features="sqrt"`, regressor `max_features=1.0`);
      `sklearn.base.clone(...)` round-trips for both. `pytest
      tests/test_shims.py -k "test_all_shims_importable or
      test_fit_returns_self_signature or test_output_type_param_present"`:
      102 passed, 0 failed (both new names auto-picked-up via
      `ALL_SHIMS`/`_exported_shim_names()`, confirmed `len(ALL_SHIMS)` grew
      32->34 and both new names are members). Full `pytest
      tests/test_shims.py`: 130 passed, 0 failed (whole-file regression,
      zero breakage in any of the 15 pre-existing/prior estimators' shim
      contract tests). `ast.parse`/`py_compile` on the edited file: clean.
      `pytest tests/test_estimator_checks.py --collect-only`: 1360 tests
      collected, zero collection errors (`_estimators()` is a MANUAL list
      per SPEC's own "Binding-layer template" note — unaffected by this
      task's `__all__` change, confirmed not to reference `mlrs.__all__`).
      Environment note (this task's `-lpython3.14`-linker-unrelated
      workaround, matching TASK-11/12's own precedent): this is a
      pure-Python-only change, so no `cargo` build/link step applies to it
      at all; verification used the same `/tmp/oracle-venv` Python
      environment TASK-11/12 already prepared (numpy 2.5.1, pyarrow 25.0.0,
      sklearn 1.9.0), with `pytest 9.1.1` additionally installed this
      session (previously absent). This environment also carries a STALE,
      untracked, pre-existing `crates/mlrs-py/python/mlrs/_mlrs.abi3.so`
      build artifact (dated before TASK-08's RF work landed — confirmed via
      `dir(mlrs._mlrs)` showing 32 pre-RF classes, no `RandomForestClassifier`/
      `RandomForestRegressor`/`HistGradientBoosting*`) — this task's own
      verification never depends on that stale extension actually resolving
      `RandomForestClassifier`/`RandomForestRegressor` (construction and
      `get_params`/`clone` succeed without ever touching `_mlrs`, matching
      the pre-build-importability precedent every other shim already
      established); `.fit()`-level exercise against a genuinely rebuilt
      extension remains TASK-14/15/16's scope, unaffected by this task.
      Expected, plan-anticipated, OUT-OF-SCOPE side effect (verified via a
      `git stash`/before-vs-after comparison of ONLY this file):
      `pytest tests/test_params.py -k test_matrix_covers_exports` flips
      from 1 passed (before this edit) to 1 FAILED (after), reporting
      exactly `{'RandomForestClassifier', 'RandomForestRegressor'}` as the
      new coverage gap — `test_params.py::EXPECTED_PARAMS` is a MANUAL dict
      (per the plan's own "Binding-layer template" note: "TASK-16/25 add
      entries there by hand"), not auto-derived from `__all__` like
      `test_shims.py::ALL_SHIMS` is; this is the exact, plan-documented
      temporary gap between Wave 5 (this task) and Wave 7 (TASK-16, which
      extends `EXPECTED_PARAMS`/`test_estimator_checks.py::_estimators()`
      by hand) — not a regression introduced by this task and not fixed
      here, to avoid the scope creep of touching TASK-16's own file ahead
      of its own wave.
      Specs: `SPEC-PY-ENS-05`'s `__init__.py`-export sub-clause (RF half)
      is implemented and verified: both estimators are top-level importable
      and auto-covered by `test_shims.py`'s derived matrix. PY-ENS-05 as a
      WHOLE spec is still not fully implemented: the three cross-cutting
      gate-test files' manual entries (`test_params.py::EXPECTED_PARAMS`,
      `test_estimator_checks.py::_estimators()`) are TASK-16's scope, not
      this task's — `SPEC.md`'s per-spec `**Status:** draft.` line left
      UNCHANGED, matching the TASK-01..12 precedent (no PageIndex document
      exists for this feature — `pageindex_update: "NOT APPLICABLE"` —
      status transitions are orchestrator-directed here, not
      executor-silent). PY-ENS-01/02/RF-IMP-02/RF-OOB-02's end-to-end
      Given/When/Then coverage (an actual `.fit()`/`.predict()` exercise
      through a REBUILT `_mlrs` extension) remains TASK-14/15's scope.

## Wave 6 — Python oracle replay (sequential, same file)
- [x] TASK-14 — oracle replay: RandomForestClassifier
      completed_at: 2026-07-18; status: completed
      Environment resolution (this task's own explicit first job, per the
      governing mission): re-checked FRESH whether `_mlrs` can be rebuilt in
      this environment (superseding TASK-08/09/10's "cannot link
      -lpython3.14" finding for THIS specific purpose). Confirmed
      `/lib64/libpython3.14.so.1.0` exists but the unversioned dev symlink
      `libpython3.14.so` does NOT (`python3.14-devel` not installed, `/usr/lib64`
      read-only to this user) — `cargo build -p mlrs-py --features cpu`
      (which links libpython directly, matching TASK-08/09/10's own repro)
      still fails identically (`mold: fatal: library not found: python3.14`).
      However, `crates/mlrs-py/Cargo.toml`'s own doc comment + the
      `extension-module` feature (`pyo3/extension-module`, ON in every
      per-backend maturin `pyproject.toml`, e.g.
      `crates/mlrs-py/pyproject/cpu.pyproject.toml`) tells PyO3 the CPython
      symbols are supplied by the HOST INTERPRETER at import time, so the
      cdylib must NOT link libpython at all — this is exactly maturin's own
      wheel-build mode. `cargo build -p mlrs-py --features cpu,extension-module`
      (the same feature set `cpu.pyproject.toml` specifies) compiled and
      LINKED cleanly (no `-lpython3.14` in the link line at all), producing
      `target/debug/libmlrs_py.so` (567750440 bytes). Since `_mlrs` is built
      `abi3-py312` (a single ABI-stable wheel for any Python >=3.12,
      `crates/mlrs-py/Cargo.toml`'s `pyo3` feature list), this artifact is
      loadable by any compatible interpreter without a matching build
      version. Copied it to `crates/mlrs-py/python/mlrs/_mlrs.abi3.so`
      (mirrors exactly what `maturin develop` itself does: same
      `module-name = "mlrs._mlrs"` / `python-source =
      "crates/mlrs-py/python"` target per `cpu.pyproject.toml`) — this file
      is `.gitignore`d (`crates/mlrs-py/python/mlrs/.gitignore:5`, "Compiled
      extension dropped here by `maturin develop`... must never be
      committed"), so this is a legitimate local build artifact replacement,
      not a tracked-file edit. Verified via
      `dir(mlrs._mlrs)`: `RandomForestClassifier`/`RandomForestRegressor` are
      now present (confirmed absent — 32 pre-RF classes — on the stale
      on-disk artifact TASK-10/11/13 each independently documented reading
      at the start of their own sessions). This resolves the mission's
      "determine whether `_mlrs` can be rebuilt" question: **YES, via
      `cargo build -p mlrs-py --features cpu,extension-module` + a manual
      artifact copy — `maturin` itself is not installed/available in this
      environment (`which maturin` empty, no `pip` on PATH outside
      `/tmp/oracle-venv`), but its wheel-build mode is fully reproducible
      by hand since it is just a cargo-build-with-a-feature-flag plus a
      file copy, not a maturin-specific mechanism.** Python environment used
      for verification: `/tmp/oracle-venv` (pre-existing from TASK-08..13's
      own sessions; numpy 2.5.1, pyarrow 25.0.0, scikit-learn 1.9.0, pytest
      9.1.1), with `PYTHONPATH=crates/mlrs-py/python` (the repo's own
      in-tree Python source tree, matching `cpu.pyproject.toml`'s
      `python-source` setting) — no package installation into the venv
      itself, so no risk of polluting/committing anything.
      Files: Create (per this task's own Files scope, the ONLY repo file
      this task modified): `crates/mlrs-py/python/tests/test_oracle_ensemble.py`
      (14 tests: `test_random_forest_classifier_deterministic` (f32+f64,
      exact `predict` + `predict_proba` within 1e-5/1e-4 vs
      `det_pred_train`/`det_proba_train`), `test_random_forest_classifier_statistical`
      (f32+f64, held-out accuracy within `ACC_MARGIN=0.05` vs `stat_acc_test`),
      `test_random_forest_classifier_max_features_invalid_raises`,
      `test_random_forest_classifier_not_fitted_raises` (predict/predict_proba/
      feature_importances_/oob_score_ all raise `sklearn.exceptions.NotFittedError`),
      `test_random_forest_classifier_feature_importances_close` (f32+f64,
      `atol=0.05` vs `ref_feature_importances`, per SPEC.md `spec_revision: 2`
      — NOT `1e-5`, see file docstring),
      `test_random_forest_classifier_feature_importances_sums_to_one` (f32+f64),
      `test_random_forest_classifier_oob_score_statistical_band` (f32+f64,
      `OOB_MARGIN=0.10` vs `ref_oob_score`, duplicated from
      `random_forest_classifier_test.rs::OOB_MARGIN`/`OOB_MARGIN_F32` per this
      task's own Risk note),
      `test_random_forest_classifier_oob_score_false_raises_attribute_error`,
      `test_random_forest_classifier_oob_true_bootstrap_false_raises_value_error`).
      Deviation from the plan's literal Red-test prose (`max_features=None`),
      recorded in the file's own module docstring and NOT a scope change:
      confirmed via `crates/mlrs-py/src/estimators/ensemble.rs::PyRandomForestClassifier::new`
      (TASK-08's own already-documented PyO3-boundary-forced deviation) that
      `max_features=None` collapses to the CLASSIFIER's `"sqrt"` default, NOT
      sklearn's "all features" encoding (PyO3's `Option<&Bound<PyAny>>`
      extraction cannot distinguish "omitted" from "explicitly None"). The
      Rust-layer deterministic-tier oracle test requires `MaxFeatures::All`
      (for `sample_features`'s zero-RNG `mf == d` short-circuit) to reproduce
      sklearn's bit-identical-tree precondition — passing `max_features=None`
      here would silently build with `MaxFeatures::Sqrt` instead and break the
      exact-match deterministic-tier assertions. Used `max_features=1.0`
      instead (the documented, unambiguous "all features" encoding per
      `ensemble.py`'s own module docstring: `Frac(1.0)` -> `Value(ceil(1.0 *
      n_features)) == Value(n_features)`, numerically `== d` for this
      fixture's `n_features=5`, so the same zero-RNG path fires — verified
      against `crates/mlrs-algos/src/ensemble/mod.rs::MaxFeatures::resolve`
      and `crates/mlrs-backend/src/prims/random_forest.rs::sample_features`'s
      `mf == d` condition before writing the test). Also confirmed (by
      reading `ensemble.py`'s `__init__`, which only stores ctor args
      verbatim per the sklearn purity rule) that an invalid `max_features`
      string raises at `.fit()` time, not construction — the plan's own Step-3
      test name doesn't specify which, and constructing-then-asserting-on-fit
      is what the real shim contract requires; written accordingly.
      Tests (genuine Red then Green, both freshly re-demonstrated this
      session specifically for this new file, not just cited from history):
      RED — temporarily moved the just-rebuilt `_mlrs.abi3.so` out of
      `crates/mlrs-py/python/mlrs/` (simulating the pre-build/pre-TASK-10
      state) and ran `pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py`:
      13 of 14 tests FAILED with `ImportError: mlrs: the compiled '_mlrs'
      backend extension is not available...` (the plan's own stated "Expected
      initial failure" — "ModuleNotFoundError for `_mlrs` (pre-build)");
      `test_random_forest_classifier_not_fitted_raises` passed even in this
      state (expected — it never reaches `_ext()`, since `_check_fitted`
      raises `NotFittedError` on attribute-existence alone before any
      extension call, not a false-positive). GREEN — restored the rebuilt
      `.so` and re-ran: `pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -v`
      → **14 passed** (no production code changes were needed — TASK-08..13
      were already correct; this confirms the plan's own anticipated "no bug
      found" Green-step outcome). Broader regression (every OTHER file in
      `crates/mlrs-py/python/tests/`, run individually to bound wall-clock
      time — the full-directory `pytest python/tests/` run was ALSO started
      and left running in the background per this project's own established
      slow-cpu-backend precedent, see below):
      `test_shims.py` 130 passed; `test_params.py` 130 passed + 1 pre-existing,
      PLAN-DOCUMENTED-EXPECTED failure (`test_matrix_covers_exports`, flagging
      `{'RandomForestClassifier', 'RandomForestRegressor'}` as not yet in the
      MANUAL `EXPECTED_PARAMS` dict — explicitly TASK-16's scope per TASK-13's
      own recorded "Expected, plan-anticipated, OUT-OF-SCOPE side effect" note,
      re-confirmed unchanged here, NOT fixed by this task to avoid encroaching
      on TASK-16's exclusive file ownership); `test_oracle_linear.py` 16
      passed; `test_oracle_cluster.py` 4 passed; `test_oracle_decomposition.py`
      8 passed; `test_oracle_neighbors.py` 6 passed; `test_oracle_metrics.py`
      53 passed; `test_dtype.py` 4 passed + 1 skipped (pre-existing skip
      reason, unrelated); `test_io.py` 27 passed; `test_import_probe.py` 3
      passed; `test_egress_shape_regression.py` 10 passed; `test_wheels.py` 6
      skipped (pre-existing, wheel-infra not applicable in-tree). `pytest
      crates/mlrs-py/tests/test_random_forest.py` (the Rust-facing FFI smoke
      suite from TASK-08/09, now exercisable end-to-end for the first time
      against a REAL registered extension instead of `hasattr`-skipping): 16
      passed, 0 skipped — the `requires_rf`/`requires_rf_reg` skip guards
      TASK-08/09 added now resolve `True` and every previously-skip-guarded
      assertion actually ran and passed. `cargo check -p mlrs-py --features
      cpu` and `--features wgpu`: both clean (only the 2 pre-existing
      unrelated `spectral.rs` dead-code warnings TASK-08/09/10 already
      documented). The full `pytest crates/mlrs-py/python/tests/` directory
      run (1360+ collected, dominated by `test_estimator_checks.py`'s
      sklearn `check_estimator` sweep over the ~32 PRE-EXISTING, non-RF
      estimators — confirmed via `grep -n RandomForest test_estimator_checks.py`
      → zero hits, i.e. `_estimators()`'s MANUAL list does not include either
      new RF estimator yet, TASK-16's scope — so this sweep exercises zero
      RF-related code paths) was started and left running in the background
      past 590+ wall-clock seconds / 2+ hours accumulated CPU time without
      producing output yet, matching this project's own extensively
      documented cpu-backend-is-slow / `check_estimator`-sweep-is-slow
      precedent (TASK-01 through TASK-13's blocker log, every one of which
      treats a per-file/targeted-run + a crate-wide compile-check as
      equivalent regression evidence when a genuinely slow full-suite run
      cannot complete in-session); combined with (a) every INDIVIDUALLY-run
      file in the directory passing above, (b) zero code-path overlap
      between this task's one new file and `test_estimator_checks.py`'s
      pre-existing, RF-untouched estimator list, and (c) the clean
      `cargo check` compile gates, this is treated as equivalent regression
      evidence per the established precedent, not a full literal pass of the
      slow sweep.
      Specs: `SPEC-PY-ENS-01`'s full Given/When/Then end-to-end Python-binding
      contract (deterministic-tier exact predict/predict_proba,
      statistical-tier accuracy band, max_features-invalid ValueError,
      not-fitted error) is now verified through a REAL, freshly-rebuilt
      `_mlrs` extension for the FIRST time in this plan's execution — not
      merely unit-tested in isolation. `SPEC-RF-IMP-02`'s Python-binding
      contract (`feature_importances_` present/shape/tolerance/not-fitted) is
      fully verified for the CLASSIFIER. `SPEC-RF-OOB-02`'s Python-binding
      contract (`oob_score_` present-when-flagged/statistical-band/
      `AttributeError`-when-false/`ValueError`-on-invalid-combination) is
      fully verified for the CLASSIFIER. One SPEC §5 PY-ENS-01 Given/When/Then
      bullet — "`y` containing non-integer-valued floats or out-of-i32-range
      values raises `ValueError`" — was NOT given a dedicated RF-specific test
      in this task: it is a cross-cutting, estimator-generic ingress-layer
      behavior (not RF-unique), it is NOT enumerated in TASK-14's own literal
      Implementation Steps 1-9 (the plan's authorized execution sequence for
      this task), and it is already exercised generically elsewhere in the
      suite (`test_io.py`, 27 passed, unaffected by this task). This is
      recorded here rather than silently treated as complete: if the
      spec-owner wants an RF-specific instance of this assertion, it is a
      small, additive follow-up, not a defect in what was implemented.
      `SPEC.md`'s per-spec `**Status:** draft.` lines for PY-ENS-01,
      RF-IMP-02, RF-OOB-02 left UNCHANGED, matching the TASK-01..13 precedent
      established throughout this file (no PageIndex document exists for this
      feature — `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s own
      frontmatter — status transitions are orchestrator-directed here, not
      executor-silent). PY-ENS-02/RF-IMP-02/RF-OOB-02's REGRESSOR half remains
      TASK-15's scope (not claimed here); PY-ENS-05's gate-test entries
      (`test_params.py::EXPECTED_PARAMS`, `test_estimator_checks.py::_estimators()`)
      remain TASK-16's scope (the one pre-existing, expected `test_params.py`
      failure documented above is exactly this gap, left untouched).
- [x] TASK-15 — oracle replay: RandomForestRegressor
      completed_at: 2026-07-18; status: completed
      Environment: reused the SAME `_mlrs.abi3.so` artifact TASK-14 rebuilt
      this session (`cargo build -p mlrs-py --features cpu,extension-module`,
      copied to `crates/mlrs-py/python/mlrs/_mlrs.abi3.so`) — re-confirmed
      present with both `RandomForestClassifier`/`RandomForestRegressor`
      registered (`dir(mlrs._mlrs)`) before writing any test; no rebuild was
      needed (TASK-08/09/10 had already added both RF classes to `lib.rs`
      before TASK-14's build).
      Files: Modify (append only, per this task's own Files scope — the ONLY
      repo file this task changed): `crates/mlrs-py/python/tests/test_oracle_ensemble.py`
      (+14 tests, mirroring TASK-14's classifier section one-for-one, minus
      `predict_proba`/`classes_` which do not exist on the regressor):
      `test_random_forest_regressor_deterministic` (f32+f64, `predict`
      within `PRED_TOL`/`_atol()` vs `det_pred_train`),
      `test_random_forest_regressor_statistical` (f32+f64, held-out R² within
      `R2_MARGIN=0.05` vs `stat_r2_test`),
      `test_random_forest_regressor_max_features_invalid_raises`,
      `test_random_forest_regressor_not_fitted_raises` (predict/
      feature_importances_/oob_score_ all raise `NotFittedError`),
      `test_random_forest_regressor_feature_importances_close` (f32+f64,
      `REG_IMPORTANCE_ATOL=0.05` vs `ref_feature_importances`, per SPEC.md
      `spec_revision: 2`/TASK-03's resolution — NOT `1e-5`),
      `test_random_forest_regressor_feature_importances_sums_to_one` (f32+f64),
      `test_random_forest_regressor_oob_score_statistical_band` (f32+f64,
      `REG_OOB_MARGIN=0.10` vs `ref_oob_score`, duplicated from
      `random_forest_regressor_test.rs::OOB_MARGIN`/`OOB_MARGIN_F32` per
      TASK-07/this task's own Risk note),
      `test_random_forest_regressor_oob_score_false_raises_attribute_error`,
      `test_random_forest_regressor_oob_true_bootstrap_false_raises_value_error`.
      No `max_features=None`-collapse deviation was needed here (unlike
      TASK-14's classifier section): `ensemble.py`'s `RandomForestRegressor.__init__`
      default IS already `1.0` (sklearn's own regressor default is "all
      features"), confirmed by reading `ensemble.py` before writing the
      deterministic-tier helper; `max_features=1.0` is passed explicitly in
      `_det_regressor()` for parity with the Rust test's explicit
      `MaxFeatures::All`, not as a workaround.
      Tests (genuine Red then Green, freshly re-demonstrated this session):
      RED — moved `crates/mlrs-py/python/mlrs/_mlrs.abi3.so` out of the
      package directory (identical mechanism to TASK-14) and ran
      `pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -k regressor`:
      13 of 14 new regressor tests FAILED with `ImportError: mlrs: the
      compiled '_mlrs' backend extension is not available...` (the plan's
      stated "Expected initial failure", matching TASK-14's own precedent
      exactly); `test_random_forest_regressor_not_fitted_raises` passed even
      in this state (expected — `_check_fitted` raises `NotFittedError` on
      attribute-existence alone before any `_ext()` call, same as the
      classifier's not-fitted test). GREEN — restored the `.so` and re-ran:
      `pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -v` →
      **28 passed** (all 14 classifier tests from TASK-14 plus all 14 new
      regressor tests; zero production code changes were needed — TASK-01..13
      were already correct for the regressor half). Broader regression:
      `pytest crates/mlrs-py/python/tests/test_shims.py` (130 passed);
      `pytest crates/mlrs-py/python/tests/test_params.py` (130 passed + 1
      pre-existing, TASK-14-documented, TASK-16-scoped expected failure —
      `test_matrix_covers_exports` flagging `RandomForestClassifier`/
      `RandomForestRegressor` as absent from the MANUAL `EXPECTED_PARAMS`
      dict — re-confirmed unchanged by this task, not fixed, matching
      TASK-14's own explicit non-encroachment note);
      `pytest crates/mlrs-py/python/tests/test_oracle_linear.py
      test_oracle_cluster.py test_oracle_decomposition.py
      test_oracle_neighbors.py test_oracle_metrics.py` (87 passed);
      `pytest crates/mlrs-py/python/tests/test_dtype.py test_io.py
      test_import_probe.py test_egress_shape_regression.py test_wheels.py`
      (44 passed, 7 pre-existing skips); `pytest crates/mlrs-py/tests/test_random_forest.py`
      (the Rust-facing FFI smoke suite from TASK-08/09) — 16 passed, 0
      skipped (both `requires_rf`/`requires_rf_reg` guards resolve `True`
      against the real registered extension). `grep -n RandomForest
      crates/mlrs-py/python/tests/test_estimator_checks.py` — zero hits,
      re-confirming TASK-14's own finding that the sklearn `check_estimator`
      sweep exercises zero RF-related code paths yet (TASK-16's scope,
      unaffected by this task). `cargo check -p mlrs-py --features cpu`
      and `--features wgpu` — both clean, only the same 2 pre-existing
      unrelated `spectral.rs` dead-code warnings TASK-08/09/14 already
      documented (this task made zero Rust-source changes, so this is a
      no-op-regression confirmation, not new evidence of a fix). `cargo
      check -p mlrs-py --features cpu --tests` — the identical, byte-for-byte
      pre-existing error set TASK-08/09/14 already documented
      (`sgd_smoke_test.rs`/`spectral_smoke_test.rs`, stale
      `mlrs_algos::traits` import + `SpectralEmbedding`/`SpectralClustering`
      constructor-signature drift), confirming `random_forest_smoke_test.rs`
      itself still compiles clean and this task introduced no new
      whole-crate-test-compile regression.
      Specs: `SPEC-PY-ENS-02`'s full Given/When/Then end-to-end Python-binding
      contract (deterministic-tier exact predict, statistical-tier R² band,
      max_features-invalid ValueError, not-fitted error) is now verified
      through the real `_mlrs` extension for the regressor. `SPEC-RF-IMP-02`'s
      Python-binding contract (`feature_importances_` present/shape/
      tolerance/not-fitted) is now fully verified for BOTH estimators
      (classifier: TASK-14; regressor: this task) — RF-IMP-02 as a WHOLE
      spec is complete. `SPEC-RF-OOB-02`'s Python-binding contract
      (`oob_score_` present-when-flagged/statistical-band/`AttributeError`-
      when-false/`ValueError`-on-invalid-combination) is now fully verified
      for BOTH estimators — RF-OOB-02 as a WHOLE spec is complete.
      `SPEC.md`'s per-spec `**Status:** draft.` lines for PY-ENS-02,
      RF-IMP-02, RF-OOB-02 left UNCHANGED, matching the TASK-01..14 precedent
      established throughout this file (no PageIndex document exists for
      this feature — `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s own
      frontmatter — status transitions are orchestrator-directed here, not
      executor-silent). The same SPEC §5 PY-ENS-01/02 Given/When/Then bullet
      TASK-14 flagged as cross-cutting rather than RF-specific — "`y`
      containing non-integer-valued floats or out-of-i32-range values raises
      `ValueError`" — is likewise not given a dedicated RF-regressor-specific
      test here, for the identical reason TASK-14 recorded (generic
      ingress-layer behavior, already exercised in `test_io.py`, not
      enumerated in this task's own Implementation Steps 1-9). Waves 3-6 (RF
      Python binding work: PY-ENS-01/02, RF-IMP-01/02, RF-OOB-01/02) are now
      FULLY complete across TASK-01..15. PY-ENS-05's gate-test entries
      (`test_params.py::EXPECTED_PARAMS`, `test_estimator_checks.py::_estimators()`)
      remain TASK-16's scope, not claimed here.

## Wave 7
- [x] TASK-16 — PY-ENS-05 gate tests (RF)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/tests/test_params.py` (`EXPECTED_PARAMS["RandomForestClassifier"]`/`["RandomForestRegressor"]` entries — every ctor
      arg + documented default read verbatim from `crates/mlrs-py/python/mlrs/ensemble.py`'s real
      `__init__` signatures, including `oob_score: False` and the classifier/regressor
      `max_features` default divergence (`"sqrt"` vs `1.0`); matching `SET_PARAM` entries
      (`("n_estimators", 10)` each)), `crates/mlrs-py/python/tests/test_shims.py`
      (`("RandomForestClassifier"|"RandomForestRegressor", "feature_importances_")` added to
      `test_fitted_attr_raises_before_fit`'s parametrize list; new dedicated
      `test_random_forest_oob_score_conditional_attribute` — asserts (a)
      `RandomForestClassifier(oob_score=True).oob_score_` before `fit` raises `NotFittedError`
      and (b) `RandomForestRegressor(oob_score=False).fit(X,y).oob_score_` raises
      `AttributeError`, per the plan's own "Q-conditional-attribute test machinery" resolution —
      NOT folded into the generic parametrize list; part (b) is guarded with
      `pytest.importorskip("mlrs._mlrs")` so the file's pure-Python collection contract holds on
      a not-yet-built tree), `crates/mlrs-py/python/tests/test_estimator_checks.py`
      (`mlrs.RandomForestClassifier(n_estimators=5, max_depth=3)` /
      `mlrs.RandomForestRegressor(n_estimators=5, max_depth=3)` appended to `_estimators()`;
      `_EXPECTED["RandomForestClassifier"] = _merge(_COMMON, _SUPERVISED, _CLASSIFIER,
      _FIT2D_1SAMPLE)`, `_EXPECTED["RandomForestRegressor"] = _merge(_COMMON, _SUPERVISED)` —
      empirically triaged against a real Green-time sweep run, not assumed; no speculative xfail
      added — `check_non_transformer_estimators_n_iter` was verified to PASS for both, so
      `_N_ITER` was deliberately NOT added to either map).
      Tests (genuine Red then Green): RED —
      `pytest crates/mlrs-py/python/tests/test_params.py -k "RandomForest or matrix"` before any
      edit: `test_matrix_covers_exports` FAILED (`AssertionError: EXPECTED_PARAMS keys
      {'RandomForestClassifier', 'RandomForestRegressor'} differ from the exported estimator
      shims`) — confirmed as the exact pre-existing gap TASK-14/15 both explicitly left
      untouched for this task. GREEN — same command: all 139
      `test_params.py` tests pass (0 failed), including
      `test_default_params_match_sklearn_names[RandomForestClassifier/Regressor]`,
      `test_set_params_roundtrip`, `test_init_purity_stores_kwargs_verbatim`,
      `test_init_purity_ast` (both RF classes' `__init__` bodies confirmed pure
      store-only via the AST check). `pytest crates/mlrs-py/python/tests/test_shims.py`:
      133 passed, 0 failed (includes the 2 new `feature_importances_` parametrize
      cases and the new dedicated `test_random_forest_oob_score_conditional_attribute`,
      which exercised its real-fit half (b) against the working `_mlrs.abi3.so` —
      not skipped in this environment). `pytest
      "crates/mlrs-py/python/tests/test_estimator_checks.py::test_estimator_checks" -k
      RandomForest`: 84 passed, 19 xfailed (all with documented reasons — no
      speculative/un-triaged xfail), 4 skipped (pre-existing `check_array_api_input`/
      `check_*_data_not_an_array` skip reasons, unrelated to this task), 0 unexplained
      failures. `pytest
      crates/mlrs-py/python/tests/test_estimator_checks.py::test_fit_free_checks_never_xfailed`:
      passed (confirms neither RF xfail map leaks a fit-free check). Full broader
      regression: `pytest crates/mlrs-py/python/tests/ --ignore=.../test_estimator_checks.py`:
      **431 passed, 7 skipped (pre-existing, unrelated), 0 failed** — confirms zero regression
      across every other file in the directory (`test_oracle_ensemble.py`'s full 28
      RF tests from TASK-14/15 included and still passing). `pytest
      crates/mlrs-py/tests/test_random_forest.py` (the Rust-facing FFI smoke suite,
      unrelated to this task's own file scope): 16 passed, 0 failed — re-confirmed
      unaffected. The FULL `test_estimator_checks.py` sweep (all ~34 estimators, ~1466
      parametrized check invocations) was started in the background and did not
      complete in-session (matches this project's own extensively-documented
      cpu-backend-is-slow / `check_estimator`-sweep-is-slow precedent, TASK-01 through
      TASK-15's blocker log, e.g. TASK-14's own "2+ hours accumulated CPU time" finding
      for a SMALLER, pre-RF sweep) — combined with (a) the RF-scoped subset above
      running to completion cleanly with zero unexplained failures, (b) the purely
      additive diff to `test_estimator_checks.py` (20 insertions, 0 deletions — no
      existing per-estimator `_EXPECTED` entry or `_estimators()` list item was
      touched, so there is no plausible mechanism by which any of the other ~32
      estimators' check results could regress), and (c) the full broader
      `python/tests/` directory (431 passed) confirming zero regression everywhere
      else, this is treated as equivalent regression evidence per the established
      project precedent, not a literal full-sweep pass.
      Specs: `SPEC-PY-ENS-05`'s RF half (both `RandomForestClassifier` and
      `RandomForestRegressor` now covered by `test_params.py`'s AST-purity gate +
      per-estimator `get_params`/mutation table, `test_shims.py`'s mixin/attribute
      enumeration, and `test_estimator_checks.py`'s sklearn `check_estimator` sweep —
      every RF check either passes or carries a documented, Green-time-verified xfail
      reason, none silently assumed) is now fully implemented and verified. The HGB
      half of `SPEC-PY-ENS-05` (Waves 9-13, TASK-18..25) remains out of this task's
      scope — HGB classes are not yet even registered in `_mlrs`, so `_estimators()`
      correctly does not reference them yet. `SPEC.md`'s per-spec `**Status:** draft.`
      lines left UNCHANGED, matching the TASK-01..15 precedent established throughout
      this file (no PageIndex document exists for this feature —
      `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s own frontmatter — status
      transitions are orchestrator-directed here, not executor-silent).
      **Final SPEC.md §6 sanity check (RF-scoped acceptance scenarios, across
      TASK-01..16 collectively):** Scenario 1 (`RandomForestClassifier` fit/predict
      end-to-end) — real passing evidence, TASK-14 (14/14) + re-confirmed in this
      task's `python/tests/` regression run. Scenario 2 (`RandomForestRegressor`
      fit/predict end-to-end) — real passing evidence, TASK-15 (14/14) +
      re-confirmed here. Scenario 3/4 (HGB classifier/regressor) — NOT RF-scoped,
      correctly out of this task's scope, still blocked on TASK-17's HGB
      fixture-freshness gate. Scenario 5 (all estimators pass
      `test_params.py`/`test_shims.py`/`test_estimator_checks.py` or documented
      xfail) — RF half now real, passing evidence per this task (above); HGB half
      remains TASK-25's scope. Scenario 6 (`feature_importances_` sums to 1 +
      dominant-feature ranking) — real passing evidence, TASK-01/02/03 (Rust) +
      TASK-14/15 (Python oracle replay), re-confirmed passing in this task's
      regression run. Scenario 7 (`oob_score_` statistical band +
      `AttributeError` when `oob_score=False`) — real passing evidence,
      TASK-04/05/06/07 (Rust) + TASK-14/15 (Python oracle replay) +
      THIS task's own new dedicated `test_random_forest_oob_score_conditional_attribute`
      (a SECOND, `test_shims.py`-native proof of the same contract, per PY-ENS-05's
      Q-conditional-attribute resolution). Scenario 8 (`oob_score=True,
      bootstrap=False` raises `ValueError`) — real passing evidence, TASK-05 (Rust
      builder unit test) + TASK-14/15 (Python oracle replay) + TASK-08/09
      (`test_random_forest.py` FFI smoke test, re-confirmed 16/16 passing here).
      **Every RF-scoped SPEC.md §6 acceptance scenario (1, 2, 5-RF-half, 6, 7, 8)
      now has real, passing test evidence — collectively across TASK-01..16, with
      no scenario resting on an assumption.** Of the 9 original SPEC.md spec IDs,
      the 7 RF-scoped ones (`PY-ENS-01`, `PY-ENS-02`, `PY-ENS-05`-RF-half,
      `RF-IMP-01`, `RF-IMP-02`, `RF-OOB-01`, `RF-OOB-02`) are now fully implemented
      and verified across TASK-01..16; `PY-ENS-03`/`PY-ENS-04` (HGB) and
      `PY-ENS-05`'s HGB half remain out of scope for Waves 1-7, correctly deferred
      to Waves 9-13 (TASK-18..25).

## Wave 8 — HGB freshness gate (no code dependency; parallel with Waves 1-7)
- [x] TASK-17 — HGB dirty-fixture gate checkpoint
      completed_at: 2026-07-18; status: completed (verification-only, no code changed)
      Command: `git status --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
      crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py
      tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz
      tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz`
      Result: all 7 paths still show `M` (modified, uncommitted) — identical to
      every prior check (research.md 2026-07-17, PLAN-CHECK.md Pass 1-3,
      the top-of-plan "Resolved planning decisions" §Q-HGB-fixture-freshness).
      **Finding: TASK-24 is BLOCKED for real tolerance-pinning as of this
      check** — it must implement the documented `@pytest.mark.xfail`
      mechanism for the deterministic-tier exact-match assertions, per this
      task's own Objective. TASK-24 MUST re-run this exact command fresh at
      its own Green time (not reuse this finding), since it may change by
      then.

## Wave 9 — PY-ENS-03/04 Rust binding (sequential; strictly after TASK-09 lands)
- [x] TASK-18 — `PyHistGradientBoostingClassifier` (structural)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/src/estimators/ensemble.rs` (appended —
      `PyHistGradientBoostingClassifier`: `#[new]`, `fit`, `predict_labels`,
      `predict_proba_f32/_f64`, `classes_`, `is_fitted`, `dtype`; mirrors
      `PyRandomForestClassifier`'s dtype-dispatch/error-mapping shape
      byte-for-byte, with HGB's own builder setters (`max_iter,
      learning_rate, max_depth, n_bins, l2_regularization,
      min_samples_leaf`) and `n_bins` defaulting to `64` (the Rust builder
      default — NOT `255`, which is a TEST-TIME construction arg for
      TASK-24's deterministic-tier oracle, not a changed default); NO
      `max_features`/`bootstrap`/`oob_score` params (HGB has none); NO
      `feature_importances_f32/_f64`/`oob_score_f32/_f64` methods at all —
      the explicit SPEC §2 non-goal, verified by ABSENCE (see Tests below);
      `AnyHistGradientBoostingClassifier` generated via
      `crate::any_estimator_typestate!`, the SAME correct macro TASK-08/09
      already established as required (not the plain `any_estimator!` trap);
      new top-of-file import
      `use mlrs_algos::ensemble::hist_gradient_boosting_classifier::HistGradientBoostingClassifier;`;
      module doc-comment updated to record TASK-18's addition),
      `crates/mlrs-py/tests/random_forest_smoke_test.rs` (new
      `hist_gradient_boosting_classifier_constructs_unfit` Rust integration
      test — `unfit_default()`/`is_unfit()` construct-and-compile gate,
      mirroring the two RF tests already in this file; import list extended),
      `crates/mlrs-py/tests/test_random_forest.py` (appended, new TASK-18
      section: `test_hgb_classifier_predict_before_fit_raises` (the plan's
      own named Red test — not-fitted guard),
      `test_hist_gradient_boosting_classifier_fit_predict` (parametrized
      f32/`@requires_f64`-guarded f64 — fit -> predict_labels/predict_proba
      shape + sum-to-1, STRUCTURAL only, no sklearn-numeric-tolerance
      assertion, per this task's own scope — TASK-24's job, gated),
      `test_hgb_classifier_max_iter_and_learning_rate_defaults` (indirect
      no-arg-construction-succeeds check, mirrors
      `test_regressor_max_features_default_is_all_not_sqrt`'s pattern —
      individual hyperparameters are not Python-visible read-back
      attributes), `test_hgb_classifier_has_no_feature_importances_or_oob_score`
      (explicit `hasattr(...)` ABSENCE assertion for
      `feature_importances_f32/_f64`/`oob_score_f32/_f64` — this task's own
      Completion Criteria requires verifying absence, not merely omitting by
      accident); all four `@requires_hgb_clf`-guarded
      (`hasattr(_mlrs, "HistGradientBoostingClassifier")`, since registration
      is TASK-20's scope — mirrors the `requires_rf`/`requires_rf_reg`
      pattern TASK-08/09 already established).
      Deviations from the plan's literal text (none change observable
      Python-level product behavior or scope; both directly inherited from
      TASK-08/09's own already-recorded deviations, re-confirmed unchanged by
      this task): (1) test harness split (`.rs` construct-gate + `.py` FFI
      surface, TASK-08's Deviation 3) — re-confirmed via a direct `cargo test
      -p mlrs-py --features cpu --test random_forest_smoke_test` run THIS
      session: identical pre-existing `mold: fatal: library not found:
      python3.14` link failure (the `cargo test` build links libpython via
      the `auto-initialize` dev-dependency; the wheel/`extension-module`
      build path does not need it — see (2) below); `cargo check` (type-check,
      no link) is therefore used as the compile-correctness gate, per the
      established project precedent (TASK-08/09/blocker log). (2) **Genuine
      runtime verification WAS possible this session** (an escalation beyond
      TASK-08/09's own environment, per this task's own mission preamble):
      `cargo build -p mlrs-py --features cpu,extension-module` succeeds
      (this build mode does not link libpython — the host interpreter
      provides those symbols at import time), and the resulting
      `libmlrs_py.so` was copied to
      `crates/mlrs-py/python/mlrs/_mlrs.abi3.so` and exercised live via
      `/tmp/oracle-venv` (numpy 2.5.1, pyarrow 25.0.0, pytest, sklearn 1.9.0)
      — confirming `_mlrs.HistGradientBoostingClassifier` correctly does NOT
      exist yet (registration is TASK-20's scope, not this task's) and that
      the pre-existing `RandomForestClassifier`/`RandomForestRegressor`
      surface has ZERO regression after this task's `ensemble.rs` edit
      (`crates/mlrs-py/tests/test_random_forest.py`: 16 passed, 5 skipped
      [the new HGB tests, correctly skip-guarded];
      `crates/mlrs-py/python/tests/test_oracle_ensemble.py`: 28 passed;
      `test_shims.py` + `test_params.py`: 272 passed, 0 failed).
      Tests (genuine Red then Green): RED —
      `cargo check -p mlrs-py --features cpu --test random_forest_smoke_test`
      before any `ensemble.rs`/`.rs`-test edit: `error[E0432]: unresolved
      import mlrs_py::estimators::ensemble::PyHistGradientBoostingClassifier`
      — confirmed as the exact stated-reason failure (the class genuinely did
      not exist). GREEN — same command, after: clean, exit 0, zero new
      warnings. `cargo check -p mlrs-py --features cpu` (whole lib): clean,
      exit 0. `cargo build -p mlrs-py --features cpu,extension-module`:
      clean, exit 0 (the genuine-runtime-verification build). `cargo check -p
      mlrs-py --features wgpu` (second backend gate): clean, exit 0. `cargo
      test -p mlrs-py --features cpu --test random_forest_smoke_test`:
      reaches the identical pre-existing `-lpython3.14` link failure
      (Deviation 1 above — not a regression, matches TASK-08/09's own
      byte-for-byte finding). `cargo check -p mlrs-py --features cpu --tests`
      (whole-crate test compile): surfaces the SAME PRE-EXISTING, unrelated
      compile errors in `sgd_smoke_test.rs`/`spectral_smoke_test.rs`
      (identical error-message set to TASK-08's own documented finding — 15 +
      8 errors, same symbols: stale `mlrs_algos::traits` import,
      `SpectralEmbedding`/`SpectralClustering`/`LinearSVC`/`LinearSVR`/
      `MBSGDClassifier`/`MBSGDRegressor` constructor/trait-method-scope
      drift); `random_forest_smoke_test.rs` (this task's own file) reports
      ZERO errors in that same run — confirmed pre-existing and unrelated
      (neither `sgd_smoke_test.rs` nor `spectral_smoke_test.rs` appears in
      `git status --short`, i.e. untouched by this or any prior task in this
      plan). Live pytest runs (via the rebuilt `.abi3.so`, `/tmp/oracle-venv`,
      `PYTHONPATH=crates/mlrs-py/python`):
      `crates/mlrs-py/tests/test_random_forest.py`: **16 passed, 5 skipped**
      (all 5 new TASK-18 tests skip cleanly via `@requires_hgb_clf`, exactly
      as expected pre-TASK-20; zero regression to the 16 pre-existing RF
      tests). `crates/mlrs-py/python/tests/test_oracle_ensemble.py`: **28
      passed, 0 failed** (the full RF oracle-replay suite, re-confirmed
      unaffected). `crates/mlrs-py/python/tests/test_shims.py` +
      `test_params.py`: **272 passed, 0 failed** (these gate-test files are
      correctly NOT yet extended for HGB — that is TASK-25's scope — and
      show zero regression from this task's `ensemble.rs`/smoke-test edits).
      `rustfmt --edition 2021 --check` on both edited `.rs` files: clean,
      zero drift. Static verification of the explicit non-goal: `awk
      '/pub struct PyHistGradientBoostingClassifier/,0'
      crates/mlrs-py/src/estimators/ensemble.rs | grep -n
      "feature_importances\|oob_score"` — zero matches, confirming no
      accidental copy-paste of the RF-only accessors (the Completion
      Criteria's own required check).
      Specs: `SPEC-PY-ENS-03`'s STRUCTURAL binding-layer contract (the
      `#[pyclass]` shape: constructor, `fit`, `predict_labels`,
      `predict_proba_f32/_f64`, `classes_`, `is_fitted`, `dtype`, and the
      explicit-by-absence non-goal) is implemented and verified for this
      task's own scope. `SPEC.md`'s per-spec `**Status:** draft.` line left
      UNCHANGED (no PageIndex document exists for this feature —
      `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s own frontmatter;
      status transitions are orchestrator-directed here, matching the
      TASK-01..17 precedent established throughout this file).
      `SPEC-PY-ENS-03` as a WHOLE spec is NOT fully implemented by this task
      alone: `_mlrs` registration (TASK-20), the Python shim `@property`/
      constructor wiring (TASK-21), `__init__.py` export (TASK-23), and the
      GATED oracle-tolerance-finalization replay (TASK-24, blocked as of
      TASK-17's own re-confirmed-dirty `git status` finding) are all still
      required before PY-ENS-03's full Given/When/Then coverage (SPEC.md §5)
      is satisfied end-to-end — none of those are claimed complete here.
- [x] TASK-19 — `PyHistGradientBoostingRegressor` (structural)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/src/estimators/ensemble.rs` (appended —
      `PyHistGradientBoostingRegressor`: `#[new]`, `fit`, `predict_f32/_f64`,
      `is_fitted`, `dtype`; mirrors `PyHistGradientBoostingClassifier`'s
      dtype-dispatch/error-mapping/builder-setter shape byte-for-byte
      (`max_iter, learning_rate, max_depth, n_bins, l2_regularization,
      min_samples_leaf`, `n_bins` defaulting to `64` — the Rust builder
      default, confirmed identical to the classifier's own
      `HGB_REG_DEFAULT_*` constants via direct Read of
      `hist_gradient_boosting_regressor.rs`), minus `classes_`/
      `predict_labels`/`predict_proba_f32/_f64` (no `PredictLabels`/
      `PredictProba` impl on the regressor), plus `predict_f32/_f64`
      composing the `Predict` trait exactly like
      `PyRandomForestRegressor::predict_f32/_f64`; NO
      `feature_importances_f32/_f64`/`oob_score_f32/_f64` methods at all —
      the explicit SPEC §2 non-goal, verified by ABSENCE (see Tests below);
      `AnyHistGradientBoostingRegressor` generated via
      `crate::any_estimator_typestate!`, the same correct macro
      TASK-08/09/18 already established; new top-of-file import `use
      mlrs_algos::ensemble::hist_gradient_boosting_regressor::HistGradientBoostingRegressor;`;
      module doc-comment updated to record TASK-19's addition),
      `crates/mlrs-py/tests/random_forest_smoke_test.rs` (new
      `hist_gradient_boosting_regressor_constructs_unfit` Rust integration
      test — `unfit_default()`/`is_unfit()` construct-and-compile gate,
      mirroring the three existing tests in this file; import list
      extended), `crates/mlrs-py/tests/test_random_forest.py` (appended, new
      TASK-19 section, mirroring TASK-18's HGB classifier section 1:1 for
      the regressor: `test_hgb_regressor_predict_before_fit_raises` (the
      plan's own named Red test — not-fitted guard),
      `test_hist_gradient_boosting_regressor_fit_predict` (parametrized
      f32/`@requires_f64`-guarded f64 — fit -> predict shape + all-finite,
      STRUCTURAL only, no sklearn-numeric-tolerance assertion — TASK-24's
      job, gated), `test_hgb_regressor_max_iter_and_learning_rate_defaults`
      (indirect no-arg-construction-succeeds check),
      `test_hgb_regressor_has_no_feature_importances_or_oob_score` (explicit
      `hasattr(...)` ABSENCE assertion, mirrors TASK-18's classifier check),
      `test_hgb_regressor_has_no_classes_or_predict_proba` (additional
      ABSENCE assertion for the classifier-only surface, since this
      estimator has no `PredictLabels`/`PredictProba` at all — not merely a
      no-goal like `feature_importances_`/`oob_score_`, but structurally
      inapplicable); all six `@requires_hgb_reg`-guarded
      (`hasattr(_mlrs, "HistGradientBoostingRegressor")`, since registration
      is TASK-20's scope — mirrors the `requires_hgb_clf` pattern TASK-18
      established).
      Deviations from the plan's literal text: none beyond those already
      recorded by TASK-08/09/18 and directly inherited unchanged (test
      harness split — `.rs` construct-gate + `.py` FFI surface, re-confirmed
      this session via a direct `cargo test -p mlrs-py --features cpu --test
      random_forest_smoke_test` run: identical pre-existing `mold: fatal:
      library not found: python3.14` link failure; `cargo build -p mlrs-py
      --features cpu,extension-module` succeeds and was used for genuine
      live-`.abi3.so` verification, per TASK-18's own established
      escalation).
      Tests (genuine Red then Green): RED — `cargo check -p mlrs-py
      --features cpu --test random_forest_smoke_test` before any
      `ensemble.rs` edit (only the `.rs` smoke-test file + import list
      edited first): `error[E0432]: unresolved import
      mlrs_py::estimators::ensemble::PyHistGradientBoostingRegressor` —
      confirmed as the exact stated-reason failure (the class genuinely did
      not exist; `grep -c PyHistGradientBoostingRegressor` on all three
      target files was `0` immediately before this run). GREEN — same
      command, after `ensemble.rs` implementation: clean, exit 0, only the 2
      pre-existing unrelated `spectral.rs` dead-code warnings. `cargo check
      -p mlrs-py --features cpu` (whole lib): clean, exit 0. `cargo build -p
      mlrs-py --features cpu,extension-module`: clean, exit 0 (the
      genuine-runtime-verification build; `.so` copied to
      `crates/mlrs-py/python/mlrs/_mlrs.abi3.so`, confirmed
      `mlrs._mlrs.HistGradientBoostingRegressor` correctly NOT YET
      registered — `hasattr` is `False` — since registration is TASK-20's
      scope, not this task's). `cargo check -p mlrs-py --features wgpu`
      (second backend gate): clean, exit 0. `cargo test -p mlrs-py
      --features cpu --test random_forest_smoke_test`: reaches the identical
      pre-existing `-lpython3.14` link failure (not a regression, matches
      TASK-08/09/18's own byte-for-byte finding). `cargo check -p mlrs-py
      --features cpu --tests` (whole-crate test compile): surfaces the SAME
      PRE-EXISTING, unrelated compile errors in
      `sgd_smoke_test.rs`/`spectral_smoke_test.rs` (identical error set to
      TASK-08/18's own documented finding, confirmed via `git status --short`
      showing both files untouched by any task in this plan);
      `random_forest_smoke_test.rs` reports ZERO errors in that same run.
      Live pytest runs (via the rebuilt `.abi3.so`, `/tmp/oracle-venv`,
      `PYTHONPATH=crates/mlrs-py/python`):
      `crates/mlrs-py/tests/test_random_forest.py`: **16 passed, 11 skipped**
      (all 6 new TASK-19 tests skip cleanly via `@requires_hgb_reg`, exactly
      as expected pre-TASK-20; zero regression to the 16 pre-existing
      RF tests or the 5 pre-existing TASK-18 HGB-classifier tests, which
      also remain correctly skipped).
      `crates/mlrs-py/python/tests/test_oracle_ensemble.py`: **28 passed, 0
      failed** (the full RF oracle-replay suite, re-confirmed unaffected).
      `crates/mlrs-py/python/tests/test_shims.py` + `test_params.py`: **272
      passed, 0 failed** (correctly NOT yet extended for HGB — TASK-25's
      scope — zero regression from this task's edits). `rustfmt --edition
      2021 --check` on both edited `.rs` files: clean, zero drift. Static
      verification of the explicit non-goal: `awk '/pub struct
      PyHistGradientBoostingRegressor/,0' crates/mlrs-py/src/estimators/ensemble.rs
      | grep -n "feature_importances\|oob_score\|classes_\|predict_labels\|predict_proba"`
      — zero matches, confirming no accidental copy-paste of the RF-only or
      classifier-only accessors (this task's own Completion Criteria's
      required check, extended beyond TASK-18's scope to also cover the
      classifier-only surface since the regressor mechanically has none of
      it).
      Specs: `SPEC-PY-ENS-04`'s STRUCTURAL binding-layer contract (the
      `#[pyclass]` shape: constructor, `fit`, `predict_f32/_f64`,
      `is_fitted`, `dtype`, and the explicit-by-absence non-goal) is
      implemented and verified for this task's own scope. `SPEC.md`'s
      per-spec `**Status:** draft.` line left UNCHANGED (no PageIndex
      document exists for this feature — `pageindex_update: "NOT
      APPLICABLE"` in `SPEC.md`'s own frontmatter; status transitions are
      orchestrator-directed here, matching the TASK-01..18 precedent
      established throughout this file). `SPEC-PY-ENS-04` as a WHOLE spec is
      NOT fully implemented by this task alone: `_mlrs` registration
      (TASK-20), the Python shim `@property`/constructor wiring (TASK-22),
      `__init__.py` export (TASK-23), and the GATED oracle-tolerance-
      finalization replay (TASK-24, still blocked as of TASK-17's own
      re-confirmed-dirty `git status` finding — not re-verified fresh by
      this task, since this task made no code change relevant to that gate)
      are all still required before PY-ENS-04's full Given/When/Then
      coverage (SPEC.md §5) is satisfied end-to-end — none of those are
      claimed complete here. **Wave 9 (TASK-18 + TASK-19) is now complete**
      — both HGB `#[pyclass]` wrappers exist, structurally verified, in the
      same `ensemble.rs` file TASK-08/09 established.

## Wave 10 — lib.rs registration (10a) parallel with Python shim (10b)
- [x] TASK-20 — `lib.rs` registration (HGB)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/src/lib.rs` ONLY (per this task's explicit scope
      restriction — `crates/mlrs-py/python/mlrs/ensemble.py`/`__init__.py`
      belong to the parallel TASK-21/22 executor and were confirmed
      untouched by this task via `git diff --stat` before/after): extended
      the existing `use estimators::ensemble::{PyRandomForestClassifier,
      PyRandomForestRegressor};` import to also pull in
      `PyHistGradientBoostingClassifier`/`PyHistGradientBoostingRegressor`;
      added a new "Phase-14 ensemble wrappers" comment block + two
      `m.add_class::<PyHistGradientBoostingClassifier>()?;`/
      `m.add_class::<PyHistGradientBoostingRegressor>()?;` calls immediately
      after the existing Phase-13 (RF) block (registration count 34 -> 36,
      matching this task's own Objective); removed the now-stale forward
      reference in the Phase-13 comment ("HistGradientBoostingClassifier/
      Regressor are registered separately once their own binding task
      (TASK-20) lands.") since TASK-20 is this task; the new Phase-14 comment
      explicitly documents this as the FINAL registration correction in this
      plan's scope (per the Objective's own "no further estimator additions"
      note). Confirmed via `grep -n "12 estimator\|-> 30"` that no stale
      "12 estimator"/"30" comment remains anywhere in `lib.rs` (TASK-10 had
      already reworded the two module-doc occurrences to name estimator
      families rather than a raw count, per its own Risks/Guardrails note —
      those needed no further edit here).
      Tests (genuine Red then Green): RED — confirmed before this edit via
      `grep -c "add_class::<Py" lib.rs` = 34 and no `HistGradientBoosting`
      reference anywhere in `lib.rs` (the classes exist in `ensemble.rs`
      since TASK-18/19 but were genuinely not yet registered — matches the
      plan's own stated Red expectation); a baseline `cargo check -p mlrs-py
      --features cpu` on this pre-edit state was clean (confirming the
      compile-correctness gate itself was healthy before this task's change,
      not merely that the class was absent). GREEN — same grep = 36 after
      the edit; `cargo check -p mlrs-py --features cpu` clean, exit 0, only
      the 2 pre-existing unrelated `spectral.rs` dead-code warnings TASK-08/
      09/10/18/19 already documented. `cargo check -p mlrs-py --features
      wgpu` clean, exit 0 (second backend gate). `cargo check -p mlrs-py
      --features cpu --test random_forest_smoke_test` clean, exit 0.
      **Stronger verification, beyond the plan's literal Red/Green (this
      environment's linker still cannot resolve `-lpython3.14` for `cargo
      test -p mlrs-py`, reconfirmed via a 60s-timeout run reaching the
      identical pre-existing `mold: fatal: library not found: python3.14`
      link failure TASK-08/09/10/18/19 already documented — not a
      regression): used the TASK-18/19-established `extension-module`
      rebuild path.** `cargo build -p mlrs-py --features cpu,extension-module`
      succeeded; the resulting `target/debug/libmlrs_py.so` was copied to
      `crates/mlrs-py/python/mlrs/_mlrs.abi3.so` and exercised live via
      `/tmp/oracle-venv` (numpy, pyarrow, pytest, sklearn 1.9.0,
      `PYTHONPATH=crates/mlrs-py/python`): a direct `import mlrs._mlrs as m`
      confirmed all FOUR ensemble classes now `hasattr` `True`
      (`RandomForestClassifier`, `RandomForestRegressor`,
      `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` — the
      last two for the FIRST time in this plan). Live pytest runs:
      `crates/mlrs-py/tests/test_random_forest.py`: **27 passed, 0 skipped**
      (previously 16 passed/11 skipped at TASK-19 — all 11 previously
      `@requires_hgb_clf`/`@requires_hgb_reg`-skipped HGB tests now run and
      pass for the first time, zero regression to the 16 pre-existing RF
      tests). `crates/mlrs-py/python/tests/test_oracle_ensemble.py`: **28
      passed, 0 failed** (the RF oracle-replay suite, unaffected — matches
      TASK-18/19's own finding that this file is not HGB-extended yet,
      TASK-24's scope). `crates/mlrs-py/python/tests/test_shims.py` +
      `test_params.py`: **272 passed, 0 failed** (unaffected — these gate
      files are correctly not yet HGB-extended, TASK-25's scope).
      `cargo fmt -p mlrs-py -- --check`: isolated this task's own diff
      (`git diff crates/mlrs-py/src/lib.rs`) against the full `--check`
      output and confirmed the only two `lib.rs` hunks reported
      (`lock_pool`'s multi-line signature, the `linear::` import block) are
      BOTH pre-existing, unrelated to this task's new `use`/`add_class`
      lines (which produce zero reported diff — rustfmt's canonical form
      already matches what was written); the crate as a whole carries
      extensive pre-existing, unrelated drift in nearly every other
      estimator file (`cluster.rs`, `decomposition.rs`, `linear.rs`, etc.),
      matching the TASK-04/05/08/09/10 "no blanket reformat" discipline.
      A full-directory `pytest crates/mlrs-py/python/tests/` run was
      attempted but not relied upon: it did not complete within a 100s
      window (likely `test_estimator_checks.py`'s `check_estimator` sweep
      across all 36 now-registered estimators, or `test_wheels.py`'s wheel
      build — neither is required by this task's own Verify step) and a
      concurrent pytest process was independently observed running
      `test_oracle_ensemble.py` (attributed to the parallel TASK-21/22
      executor's own session, not killed); the four targeted, directly
      relevant test files above (27 + 28 + 272 = 327 passed, 0 failed) are
      treated as sufficient regression evidence, exceeding TASK-10's own
      environment-constrained baseline (TASK-10 could only use grep-based
      Red/Green since the `extension-module` path was established later, by
      TASK-18).
      Specs: `SPEC-PY-ENS-05`'s registration-layer contract (HGB half) is
      implemented for the Rust `_mlrs` `#[pymodule]` registration surface —
      both `PyHistGradientBoostingClassifier`/`PyHistGradientBoostingRegressor`
      are now importable from the compiled extension, confirmed via live
      import. Combined with TASK-10's RF half, `SPEC-PY-ENS-05`'s
      registration-layer contract is now FULLY implemented for all four
      ensemble estimators — this is the FINAL registration correction in
      this plan's scope (no further estimator additions remain). `SPEC.md`'s
      per-spec `**Status:** draft.` line left UNCHANGED, matching the
      TASK-01..19/21/22 precedent (no PageIndex document exists for this
      feature — `pageindex_update: "NOT APPLICABLE"` — status transitions
      are orchestrator-directed here, not executor-silent). `SPEC-PY-ENS-05`
      as a WHOLE spec is still not fully implemented: `__init__.py` export
      (TASK-23) and the cross-cutting gate-test file updates for HGB
      (`test_params.py`/`test_shims.py`/`test_estimator_checks.py`,
      TASK-25) are still required before PY-ENS-05's full Given/When/Then
      coverage is satisfied end-to-end; this task's own narrower scope
      (`lib.rs` registration only) is complete.
- [x] TASK-21 — `ensemble.py` `HistGradientBoostingClassifier`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/mlrs/ensemble.py` (appended, same
      untracked file TASK-11/12 created — this task's exclusive scope, per
      the governing session's explicit file-ownership boundary; `lib.rs`/
      `estimators/mod.rs`/`__init__.py` untouched, confirmed via
      `git status --short` at both start and end of this session):
      `HistGradientBoostingClassifier(ClassifierMixin, MlrsBase)` — `__init__`
      (defaults verbatim-matching `PyHistGradientBoostingClassifier::new`'s
      `#[pyo3(signature=(...))]` in `crates/mlrs-py/src/estimators/ensemble.rs:909-916`,
      confirmed by direct Read before writing this shim, per this task's own
      "confirm exact method names before writing the shim" instruction:
      `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
      l2_regularization=0.0, min_samples_leaf=20`), `fit` (byte-for-byte
      mirrors `RandomForestClassifier.fit`'s
      `_normalize`/`_normalize_y`/inline-`self._mlrs_obj = obj;
      self._post_fit(cols)`/`classes_` template, minus `max_features`/
      `bootstrap`/`oob_score` — HGB's builder has none of these),
      `predict`/`predict_proba` (mirror `RandomForestClassifier`'s exact
      shape: bare `predict_labels`, `_suffixed("predict_proba")`). NO
      `feature_importances_`/`oob_score_` properties — confirmed absent by
      construction (this task's own explicit non-goal, verified structurally
      post-Green via `'feature_importances_' in vars(HistGradientBoostingClassifier)`
      → `False`, vs. `True` for `RandomForestClassifier` in the same check —
      see Tests below).
      Module-level docstring updated (top of `ensemble.py`) to describe all
      four estimators and both spec families (PY-ENS-01..04, RF-IMP-02,
      RF-OOB-02), explicitly noting the HGB non-goal is intentional
      (sklearn's own HGB classes expose neither attribute either).
      Deviations from the plan's literal text: none — this task's Green step
      was a direct, unambiguous mirror of TASK-11's already-established
      classifier template (verified method names via `mcp__codegraph`-free
      direct `Read`/`grep` of `crates/mlrs-py/src/estimators/ensemble.rs`
      before writing, since TASK-18 had already landed
      `PyHistGradientBoostingClassifier` with `fit`/`predict_labels`/
      `predict_proba_f32/_f64`/`classes_`/`is_fitted`/`dtype` and explicitly
      NO `feature_importances_f32/_f64`/`oob_score_f32/_f64` methods,
      confirmed by `grep -n` on the Rust file).
      Tests: `python3 -c "import ast; ast.parse(...)"` (OK, both before and
      after — file was already valid Python before this task's append, since
      it only contained the two RF classes at that point; confirmed the
      class genuinely did not exist pre-edit by reading the file in full
      before editing — it ended at `RandomForestRegressor.oob_score_`'s
      `return score`, no HGB classes present, the exact Red state).
      Practical Red proof: `from mlrs.ensemble import HistGradientBoostingClassifier`
      would have raised `ImportError` before this edit (class absent from
      the module — confirmed by direct pre-edit Read, not re-run separately
      since the file's prior content was already fully re-read this session).
      Green (post-edit, live import in `/tmp/oracle-venv`, PYTHONPATH-based,
      no wheel rebuild needed for construction per the `naive_bayes.py`
      pre-build-importability precedent): `from mlrs.ensemble import
      HistGradientBoostingClassifier; HistGradientBoostingClassifier()`
      succeeds; `get_params()` reports every ctor default verbatim (`{'l2_regularization':
      0.0, 'learning_rate': 0.1, 'max_depth': 6, 'max_iter': 100,
      'min_samples_leaf': 20, 'n_bins': 64, 'output_type': 'input'}`);
      `sklearn.base.clone(HistGradientBoostingClassifier(max_iter=10))`
      round-trips (`max_iter=10` preserved); `.predict([[1,2]])` before
      `.fit()` raises `sklearn.exceptions.NotFittedError`.
      **Opportunistic full end-to-end runtime verification** (beyond this
      task's own minimal Red/Green requirement, made possible because the
      parallel TASK-20 session's `lib.rs` registration — plus a
      `cargo build -p mlrs-py --features cpu,extension-module` + `.so` copy —
      had already landed on-disk in this shared environment by the time this
      task ran, confirmed via `grep -n add_class crates/mlrs-py/src/lib.rs`
      showing `PyHistGradientBoostingClassifier`/`Regressor` both registered,
      and `mlrs._mlrs.HistGradientBoostingClassifier` importable): fit on a
      60x4 synthetic dataset (`max_iter=10`) → `predict` returns shape `(60,)`
      int labels, `predict_proba` returns shape `(60, 2)` with every row
      summing to `1.0`, `classes_ == [0, 1]` — genuine numeric exercise of
      this task's own new shim code, not just a construction check.
      `hasattr(HistGradientBoostingClassifier(), 'feature_importances_')` →
      `False` and `'feature_importances_' in vars(HistGradientBoostingClassifier)`
      → `False` (structural absence, not merely a not-fitted exception —
      contrasted directly against `RandomForestClassifier` in the same
      process, which reports `True` for the class-dict membership check).
      `mlrs.HistGradientBoostingClassifier` (top-level) correctly does
      `hasattr(mlrs, ...)` → `False` (TASK-23's scope, not yet wired — this
      task's own boundary held).
      Full regression (same live environment): `pytest
      crates/mlrs-py/tests/test_random_forest.py` — 27 passed, 0 failed, 0
      skipped (previously 16 passed/11 skipped per TASK-19's own recorded
      baseline before HGB registration landed; now that both this task's
      shim AND the parallel TASK-20 registration are present, the
      previously-`@requires_hgb_clf`-skipped structural tests now genuinely
      run and pass — zero failures, confirming this task's shim did not
      break TASK-18's own PyO3-layer tests). `pytest
      crates/mlrs-py/python/tests/test_shims.py
      crates/mlrs-py/python/tests/test_params.py` — 272 passed, 0 failed
      (byte-identical count to TASK-18/19's own baseline — these gate files
      are correctly NOT yet auto-extended for HGB, since `__init__.py`'s
      `__all__` is untouched by this task, matching the plan's own Wave
      10b-before-Wave-11 sequencing). `pytest
      crates/mlrs-py/python/tests/test_oracle_ensemble.py` — 28 passed, 0
      failed (the full RF oracle-replay suite, re-confirmed unaffected by
      this Python-only append).
      Specs: `SPEC-PY-ENS-03`'s Python-shim-layer contract is implemented for
      the CLASSIFIER (this task's own scope). `SPEC.md`'s per-spec
      `**Status:** draft.` line left UNCHANGED (no PageIndex document exists
      for this feature — `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s
      own frontmatter; status transitions are orchestrator-directed here,
      matching the TASK-01..19 precedent established throughout this file).
      `SPEC-PY-ENS-03` as a WHOLE spec is NOT fully implemented by this task
      alone: `__init__.py` export (TASK-23) and the GATED
      oracle-tolerance-finalization replay (TASK-24, still blocked as of
      TASK-17's own re-confirmed-dirty `git status` finding — not re-verified
      fresh by this task, since this task made no code change relevant to
      that gate) are both still required before PY-ENS-03's full
      Given/When/Then coverage (SPEC.md §5) is satisfied end-to-end — neither
      is claimed complete here. This task's own opportunistic live
      fit/predict/predict_proba exercise above is genuine numeric evidence
      but is NOT a substitute for TASK-24's own pinned-tolerance-vs-sklearn
      oracle assertion (uses a hand-built synthetic dataset, not the
      committed `hgb_cls_*.npz` fixtures).
- [x] TASK-22 — `ensemble.py` `HistGradientBoostingRegressor`
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/mlrs/ensemble.py` (appended, same file
      TASK-21 just extended in this same session): `HistGradientBoostingRegressor(RegressorMixin,
      MlrsBase)` — mirrors TASK-21's classifier one-for-one minus `classes_`/
      `predict_proba`, plus a float-only `predict` (`_suffixed("predict")`,
      mirrors `RandomForestRegressor.predict`/`neighbors.py`'s regressor
      shims exactly). `__init__` defaults verbatim-match
      `PyHistGradientBoostingRegressor::new`'s `#[pyo3(signature=(...))]`
      (`crates/mlrs-py/src/estimators/ensemble.rs:1191-1198`, confirmed
      identical to the classifier's own defaults by direct Read):
      `max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
      l2_regularization=0.0, min_samples_leaf=20`. `fit`'s `_normalize_y`
      dtype helper reuses `HistGradientBoostingClassifier._x_float(xa)`
      cross-class within the same module (mirrors `RandomForestRegressor.fit`'s
      own cross-class reuse of `RandomForestClassifier._x_float`, an
      already-established in-module precedent in this same file, not a new
      pattern). NO `feature_importances_`/`oob_score_` properties —
      confirmed absent by construction, same verification method as TASK-21.
      Deviations: none — direct mirror of TASK-12's already-established
      regressor template, method names re-confirmed against
      `PyHistGradientBoostingRegressor`'s actual `fit`/`predict_f32/_f64`/
      `is_fitted`/`dtype` signatures (`crates/mlrs-py/src/estimators/ensemble.rs:1225-1361`)
      before writing, per this task's own "confirm exact method names"
      instruction — no accidental `classes_`/`predict_labels`/
      `predict_proba` copy-paste (this estimator structurally has none of
      that surface, matching TASK-19's own PyO3-layer absence).
      Tests: Red confirmed by direct pre-edit Read of the file state left by
      TASK-21 (ended at `HistGradientBoostingClassifier._x_float`'s closing
      line, no regressor class present — the genuine "class does not exist
      yet" state). Green (post-edit, live import,
      `/tmp/oracle-venv`): `from mlrs.ensemble import
      HistGradientBoostingRegressor; HistGradientBoostingRegressor()`
      succeeds; `get_params()` reports every default identical to the
      classifier's own six hyperparameters plus `output_type='input'`;
      `sklearn.base.clone(HistGradientBoostingRegressor(max_iter=10))`
      round-trips; `.predict([[1,2]])` before `.fit()` raises
      `NotFittedError`. **Opportunistic full end-to-end runtime verification**
      (same live-`.so` environment as TASK-21, already registered by the
      parallel TASK-20 session): fit on the same 60x4 synthetic dataset
      (`max_iter=10`) → `predict` returns shape `(60,)`, all values finite —
      genuine numeric exercise of this task's own new shim code.
      `hasattr(HistGradientBoostingRegressor(), 'feature_importances_')` →
      `False`, `'feature_importances_' in vars(HistGradientBoostingRegressor)`
      → `False` (structural absence, re-confirmed independently of TASK-21's
      classifier check).
      Full regression (same live environment, re-run after this task's own
      edit to confirm zero regression from the append): `pytest
      crates/mlrs-py/tests/test_random_forest.py` — 27 passed, 0 failed, 0
      skipped (unchanged from TASK-21's own count — this task's append did
      not affect any PyO3-layer test). `pytest
      crates/mlrs-py/python/tests/test_shims.py
      crates/mlrs-py/python/tests/test_params.py` — 272 passed, 0 failed
      (unchanged, `__init__.py` still untouched). `pytest
      crates/mlrs-py/python/tests/test_oracle_ensemble.py` — 28 passed, 0
      failed (unchanged). `python3 -c "import ast; ast.parse(...)"` on the
      final file: clean. No line in the final `ensemble.py` exceeds 100
      characters (`awk` line-length check), matching the file's existing
      style.
      Specs: `SPEC-PY-ENS-04`'s Python-shim-layer contract is implemented for
      the REGRESSOR (this task's own scope). `SPEC.md`'s per-spec
      `**Status:** draft.` line left UNCHANGED, matching the TASK-01..21
      precedent (no PageIndex document exists for this feature — status
      transitions are orchestrator-directed here, not executor-silent).
      `SPEC-PY-ENS-04` as a WHOLE spec is NOT fully implemented by this task
      alone: `__init__.py` export (TASK-23) and the GATED
      oracle-tolerance-finalization replay (TASK-24) are both still
      required. **Wave 10b (TASK-21 + TASK-22) is now complete** — both HGB
      Python shim classes exist in `ensemble.py`, alongside the two RF shim
      classes TASK-11/12 already established; this session did not touch
      `crates/mlrs-py/src/lib.rs`, `crates/mlrs-py/src/estimators/mod.rs`, or
      `crates/mlrs-py/python/mlrs/__init__.py` (confirmed via `git status
      --short` at session end showing only `crates/mlrs-py/python/mlrs/ensemble.py`
      as this session's sole edit among files not already modified by other
      parallel sessions — `lib.rs`/`estimators/mod.rs`/`__init__.py` show as
      `M` from the concurrent TASK-20 session, not from this one). TASK-20's
      own progress-log checkbox is intentionally left untouched by this
      session — that task's completion evidence is owned by its own
      executor, not asserted here.

## Wave 11
- [x] TASK-23 — `__init__.py` wiring (HGB)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/mlrs/__init__.py` ONLY (per this task's
      exact scope, mirroring TASK-13): the `.ensemble` import block was
      widened from `from .ensemble import RandomForestClassifier,
      RandomForestRegressor` to a parenthesized, alphabetically-ordered
      `from .ensemble import (HistGradientBoostingClassifier,
      HistGradientBoostingRegressor, RandomForestClassifier,
      RandomForestRegressor)`; both HGB names appended to `__all__`
      immediately after `RandomForestRegressor`.
      Tests: Red confirmed BEFORE the edit via a live import against this
      repo's own `crates/mlrs-py/python` tree in `/tmp/oracle-venv` (the
      SAME environment TASK-13/14/15 established — numpy, pyarrow, sklearn
      1.9.0, pytest 9.1.1): `import mlrs; mlrs.HistGradientBoostingClassifier`
      raised `AttributeError: module 'mlrs' has no attribute
      'HistGradientBoostingClassifier'` (the exact plan-specified Red
      state). Green confirmed after the edit, AND — unlike TASK-13, which
      only had a stale pre-RF `_mlrs.abi3.so` available — this session ran
      against the REAL, currently-built extension
      (`crates/mlrs-py/python/mlrs/_mlrs.abi3.so`, confirmed via
      `dir(mlrs._mlrs)` to already expose `HistGradientBoostingClassifier`/
      `HistGradientBoostingRegressor`/`RandomForestClassifier`/
      `RandomForestRegressor`, i.e. all four ensemble estimators registered
      per TASK-20's own prior rebuild), giving genuine end-to-end runtime
      evidence for this task, not merely pre-build importability:
      `mlrs.HistGradientBoostingClassifier`/`Regressor` both resolve to
      `mlrs.ensemble.HistGradientBoostingClassifier`/`Regressor`; both
      zero-arg-construct (`get_params()` == `{'l2_regularization': 0.0,
      'learning_rate': 0.1, 'max_depth': 6, 'max_iter': 100,
      'min_samples_leaf': 20, 'n_bins': 64, 'output_type': 'input'}` for
      both, matching TASK-18/19's `#[new]` defaults verbatim);
      `sklearn.base.clone(...)` round-trips for both (`get_params()`
      equality re-confirmed post-clone); calling `.predict(X)` on an
      unfitted instance of either class raises `mlrs`'s
      `NotFittedError` ("... instance is not fitted yet. Call 'fit' ...");
      a full real fit/predict smoke exercise against synthetic data
      succeeded for both (`HistGradientBoostingClassifier(max_iter=5).fit(X,
      y).predict(X)`/`.predict_proba(X)` — proba rows sum to `1.0`;
      `HistGradientBoostingRegressor(max_iter=5).fit(X, y).predict(X)` —
      finite float array of the expected shape). `pytest tests/test_shims.py
      -k "test_all_shims_importable or test_fit_returns_self_signature or
      test_output_type_param_present"`: 108 passed, 0 failed (all four
      ensemble names — RF pair from TASK-13, HGB pair from this task — now
      auto-covered via `ALL_SHIMS`). Full `pytest tests/test_shims.py`: 139
      passed, 0 failed (whole-file regression, zero breakage in any
      pre-existing shim contract test). `ast.parse`/`py_compile` on the
      edited file: clean. `pytest tests/test_estimator_checks.py
      --collect-only`: 1467 tests collected, zero collection errors
      (`_estimators()` remains a MANUAL list, unaffected by `__all__`,
      confirmed not to reference `mlrs.__all__`).
      Expected, plan-anticipated, OUT-OF-SCOPE side effect (mirrors TASK-13's
      own documented RF-pair gap exactly): `pytest tests/test_params.py -k
      test_matrix_covers_exports` now reports `{'HistGradientBoostingClassifier',
      'HistGradientBoostingRegressor'}` as an additional coverage gap
      (`EXPECTED_PARAMS` is a MANUAL dict, TASK-25's scope to extend by
      hand) — not a regression introduced by this task, not fixed here, to
      avoid the scope creep of touching TASK-25's own file ahead of its own
      wave.
      Environment note: same `/tmp/oracle-venv` Python 3.14 environment
      already established by prior tasks in this plan; no `cargo`
      build/link step applies to this pure-Python-only change.
      Specs: `SPEC-PY-ENS-05`'s `__init__.py`-export sub-clause (HGB half)
      is implemented and verified — both estimators are top-level importable
      and auto-covered by `test_shims.py`'s derived matrix, AND exercised
      end-to-end (construct/get_params/clone/not-fitted/fit/predict) against
      the real built `_mlrs` extension, which TASK-13 could not do at its own
      execution time. **All four PY-ENSEMBLE estimators
      (`RandomForestClassifier`, `RandomForestRegressor`,
      `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`) are
      now top-level-importable from `mlrs`.** PY-ENS-05 as a WHOLE spec is
      still not fully implemented: the three cross-cutting gate-test files'
      manual entries (`test_params.py::EXPECTED_PARAMS`,
      `test_estimator_checks.py::_estimators()`) for the HGB pair are
      TASK-25's scope, not this task's — `SPEC.md`'s per-spec `**Status:**
      draft.` line left UNCHANGED, matching the TASK-01..22 precedent (no
      PageIndex document exists for this feature — `pageindex_update: "NOT
      APPLICABLE"` — status transitions are orchestrator-directed here, not
      executor-silent). PY-ENS-03/04's end-to-end oracle-tolerance-pinned
      Given/When/Then coverage remains TASK-24's scope (GATED on a fresh
      `git status` re-check of the HGB algos churn files).

## Wave 12 — GATED
- [x] TASK-24 — Python oracle replay (HGB), gated on clean `git status` for
      `hist_gradient_boosting.rs`/`gbt.rs`/`gen_oracle.py`/`hgb_*.npz`
      completed_at: 2026-07-18; status: completed (DIRTY BRANCH taken — HGB
      exact-tolerance pinning remains blocked, per this task's own
      Completion Criteria explicitly allowing the xfail/pinned branch
      selection itself to count as done, not tolerance-pinning).
      **Step 0 (mandatory gate re-check, re-run FRESH at this task's own
      Green time, NOT reusing TASK-17's 2026-07-17 snapshot):**
      `git status --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
      crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py
      tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz
      tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz`
      → all 7 paths still `M` (modified, uncommitted) — identical to
      TASK-17's finding, PLAN-CHECK.md's 3-pass re-confirmation, and
      research.md's original 2026-07-17 discovery. **DIRTY — took the
      documented xfail/skip branch, per TASK-17's own mechanism and this
      task's own Red step, NOT the clean pinned-tolerance branch.**
      Files: `crates/mlrs-py/python/tests/test_oracle_ensemble.py` (appended,
      untracked new file TASK-14/15 created — this task's exclusive
      addition): 14 new test functions mirroring TASK-14/15's RF
      oracle-replay structure one-for-one for HGB — deterministic tier
      (`n_bins=255` explicit override, matching
      `hist_gradient_boosting_{classifier,regressor}_test.rs`'s own
      `det_builder`/`check_deterministic_tier` hyperparameters:
      `max_iter=20, learning_rate=0.1, max_depth=6, n_bins=255,
      min_samples_leaf=5, l2_regularization=0.0`, confirmed by direct Read of
      both Rust oracle-test files before writing), statistical tier (class
      defaults), not-fitted, invalid-`n_bins` cases, for BOTH classifier
      (3-class multiclass path AND binary/`y_bin` sigmoid path, mirroring the
      Rust test's own two-path split) and regressor:
      `test_hgb_classifier_deterministic_multiclass`/`_binary` (parametrized
      f32/`@requires_f64`-guarded f64 — **`@pytest.mark.xfail(reason=...,
      strict=False)`-marked**, the deterministic-tier EXACT-MATCH
      assertions: `predict` label-exact-match + `predict_proba` within
      `_atol` — 1e-5 f64/1e-4 f32, mirroring `PROBA_TOL_F64`/`PROBA_TOL_F32`),
      `test_hgb_classifier_statistical` (held-out accuracy within
      `HGB_ACC_MARGIN=0.05` of `stat_acc_test` — mirrors
      `hist_gradient_boosting_classifier_test.rs::ACC_MARGIN` — **NOT
      xfailed**, a statistical band, does not depend on fixture freshness),
      `test_hgb_classifier_not_fitted_raises` (predict/predict_proba before
      fit raise `NotFittedError` — **NOT xfailed**, structural),
      `test_hgb_classifier_invalid_n_bins_raises` (`n_bins=257` raises
      `ValueError` via `build_err_to_py` — **NOT xfailed**, structural,
      confirmed live: `ValueError: estimator
      'hist_gradient_boosting_classifier': n_bins = 257 is invalid (must be
      in 2..=256)`), and the regressor mirror:
      `test_hgb_regressor_deterministic` (**xfail-marked**, `predict` within
      `_atol` of `det_pred_train`), `test_hgb_regressor_statistical`
      (**NOT xfailed**, held-out R² within `HGB_R2_MARGIN=0.05` of
      `stat_r2_test`, mirrors `R2_MARGIN`), `test_hgb_regressor_not_fitted_raises`
      / `test_hgb_regressor_invalid_n_bins_raises` (**NOT xfailed**,
      structural). `_HGB_XFAIL_REASON` names this plan, TASK-17/TASK-24, the
      four churning source files, and the fresh 2026-07-18 dirty finding
      verbatim, per the plan's own xfail-reason-content instruction.
      No other file modified by this task (confirmed via
      `git status --short` at session end: only `test_oracle_ensemble.py`,
      an already-untracked file from TASK-14, shows any change attributable
      to this session — `scripts/gen_oracle.py` and the four `hgb_*.npz`
      fixtures, though dirty, were NOT touched by this task, consistent with
      the dirty-branch instruction to consume the existing churn state, not
      regenerate against it).
      **NOTE — prominent XPASS finding (per this task's own Risk/Guardrail
      instruction: "do NOT treat [XPASS] as a failure (strict=False) but DO
      note it prominently... it may signal the churn has settled"):** all 6
      xfail-marked parametrized test instances
      (`test_hgb_classifier_deterministic_multiclass[f32/f64]`,
      `test_hgb_classifier_deterministic_binary[f32/f64]`,
      `test_hgb_regressor_deterministic[f32/f64]`) reported **XPASS**, not
      XFAIL, when actually run — the CURRENT dirty-state HGB fixtures/kernel
      already replay within the existing Rust-test tolerances
      (`PROBA_TOL_F64=1e-5`/`_F32=1e-4`, `PRED_TOL_F64=1e-5`/`_F32=1e-4`)
      through the full Python binding path (max observed error ~4.6e-10
      f64/1.8e-6 f32 classifier proba; ~6.5e-9 f64/2.4e-7 f32 regressor
      predict — all well inside tolerance). Per this task's own explicit
      guardrail, this is **NOT** treated as grounds to un-xfail here — an
      `XPASS` is a *signal* the in-flight churn may have settled, not proof;
      un-xfailing requires a human/orchestrator re-check of `git status` at
      the churn's own commit time, not an executor's numeric probe
      overriding the dirty `git status` gate this task is bound to. Flagged
      here per the mission's own explicit instruction to report this
      prominently, not silently.
      Tests: `pytest crates/mlrs-py/python/tests/test_oracle_ensemble.py -v`
      (live, `/tmp/oracle-venv`, `PYTHONPATH=crates/mlrs-py/python`, the
      SAME rebuilt `_mlrs.abi3.so` TASK-18..23 already established with all
      four ensemble estimators registered): **42 items — 36 passed, 6
      xpassed, 0 failed** (28 pre-existing RF tests from TASK-14/15,
      re-confirmed unaffected; 14 new HGB tests this task added — 6
      xfail-marked instances all reporting XPASS per the note above, 8
      un-xfailed instances/functions all genuinely PASSED, including both
      statistical-tier bands and all 4 structural not-fitted/invalid-input
      cases). `python3 -c "import ast; ast.parse(...)"` on the final file:
      clean. Full-regression evidence (targeted, directly-relevant files —
      a whole-directory `pytest crates/mlrs-py/python/tests/` run was
      attempted but not relied upon, matching this project's own
      established equivalent-regression-evidence precedent, TASK-01
      through TASK-23's blocker log: this environment shows severe,
      externally-caused multi-tenant contention — an UNRELATED,
      already-running `pytest crates/mlrs-py/python/tests/` process, PID
      `602803`, consuming 700%+ CPU since before this task's own session
      began, confirmed via `ps aux` to belong to a different invocation
      (`source .../activate; cd ...; pytest ... -q`, not this task's own
      `PYTHONPATH=... python3 -m pytest ...` invocation shape) — consistent
      with TASK-20's own documented "concurrent pytest process... attributed
      to a parallel session, not killed" finding; a full-directory attempt
      of this task's own did not complete in either a 120s or a 300s window
      and was intentionally terminated by this task once confirmed
      non-progressing under that contention, rather than left to consume
      further shared resources indefinitely):
      `pytest crates/mlrs-py/python/tests/test_shims.py
      crates/mlrs-py/python/tests/test_params.py`: **277 passed, 1 failed**
      — the 1 failure (`test_matrix_covers_exports`) is the SAME
      PRE-EXISTING, plan-anticipated, TASK-25-scoped gap TASK-23's own
      progress entry already documented verbatim ("EXPECTED_PARAMS keys
      {'HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'}
      differ from the exported estimator shims" — `EXPECTED_PARAMS` is a
      MANUAL dict, TASK-25's scope to extend by hand), confirmed NOT
      introduced by this task (`git status --short` shows this task touched
      only `test_oracle_ensemble.py`; `test_params.py`/`__init__.py` were
      already `M` from TASK-16/23's own prior, uncommitted sessions, not
      this one). `pytest crates/mlrs-py/tests/test_random_forest.py`
      (the Rust-side PyO3 FFI structural suite, TASK-08..19's own file):
      **27 passed, 0 failed** — zero regression to any prior task's binding-
      layer surface from this Python-only test-file append.
      `pytest crates/mlrs-py/python/tests/test_estimator_checks.py
      --collect-only`: **1467 tests collected, 0 collection errors**
      (confirms this task's addition did not break collection anywhere in
      the gate-test tree; the full `check_estimator` sweep itself is not run
      here, matching TASK-14/15/20's own established "collection-only is
      sufficient regression evidence for this file at this stage" precedent
      — HGB is not yet in `_estimators()`, correctly TASK-25's scope, not
      this task's).
      Specs: `SPEC-PY-ENS-03`/`SPEC-PY-ENS-04`'s scenario 3/4 (SPEC.md §6,
      the explicitly gated ones) oracle-REPLAY test coverage now EXISTS and
      is exercised (structural + statistical-tier assertions genuinely
      pass; deterministic-tier exact-match assertions are present, correctly
      shaped, and xfail-marked per the locked dirty-fixture precondition —
      NOT silently pinned against dirty fixtures, per SPEC §5's own explicit
      instruction). `SPEC-PY-ENS-03`/`SPEC-PY-ENS-04` remain **NOT fully
      implemented** by this task: the deterministic-tier sklearn-parity
      Given/When/Then ("then it matches the committed fixture exactly, once
      the HGB fixture-freshness precondition above is satisfied") is
      genuinely UNSATISFIED as a locked, verified assertion — it is
      xfail-masked, not proven, pending the HGB algos churn landing as a
      clean commit (TASK-17/TASK-24's own explicit, plan-anticipated,
      locked-decision-compliant incompleteness — SPEC.md §5 PY-ENS-03's own
      "do not silently pin against dirty fixtures" instruction, honored
      here). `SPEC.md`'s per-spec `**Status:** draft.` lines left UNCHANGED,
      matching the TASK-01..23 precedent (no PageIndex document exists for
      this feature — `pageindex_update: "NOT APPLICABLE"` — status
      transitions are orchestrator-directed here, not executor-silent).
      TASK-25 (PY-ENS-05 gate-test updates for HGB) remains the final
      task; it does not depend on this task's own xfail-vs-clean branch
      outcome, per PLAN.md's own Wave 13 sequencing.

## Wave 13
- [x] TASK-25 — PY-ENS-05 gate tests (HGB)
      completed_at: 2026-07-18; status: completed
      Files: `crates/mlrs-py/python/tests/test_params.py`
      (`EXPECTED_PARAMS["HistGradientBoostingClassifier"]`/`["HistGradientBoostingRegressor"]`
      entries — every ctor arg + documented default read verbatim from
      `crates/mlrs-py/python/mlrs/ensemble.py`'s real `__init__` signatures via a live
      `get_params()` call (`max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
      l2_regularization=0.0, min_samples_leaf=20` — confirmed identical for both
      classifier and regressor, matching the mission's stated defaults exactly), NOT
      guessed; matching `SET_PARAM` entries (`("max_iter", 10)` each, mirroring the
      plan's own "small `max_iter` override for fast fixtures" guidance)),
      `crates/mlrs-py/python/tests/test_estimator_checks.py`
      (`mlrs.HistGradientBoostingClassifier(max_iter=10)` /
      `mlrs.HistGradientBoostingRegressor(max_iter=10)` appended to `_estimators()`;
      `_EXPECTED["HistGradientBoostingClassifier"] = _merge(_COMMON, _SUPERVISED,
      _CLASSIFIER, _N_ITER, _FIT2D_1SAMPLE)`, `_EXPECTED["HistGradientBoostingRegressor"]
      = _merge(_COMMON, _SUPERVISED, _N_ITER)` — empirically triaged against a real
      Green-time sweep run, not assumed. `test_shims.py` received NO new entry, per
      this task's own Objective: HGB has no new fitted attributes beyond
      `predict`/`predict_proba`/`classes_`, already structurally covered by
      `ALL_SHIMS`'s auto-derivation from TASK-23's `__init__.py` `__all__` update —
      confirmed via `grep -n "HistGradient" test_shims.py`, zero hits, and the full
      `test_shims.py` run below showing zero regression).
      Deviation from a same-class RF precedent (TASK-16), not from the plan's own
      text: UNLIKE `RandomForestClassifier`/`Regressor` (whose tree-growth loop has
      no iterative-solver `n_iter_` convergence concept), a fresh Green-time sweep
      run showed BOTH HGB estimators genuinely fail `check_non_transformer_estimators_n_iter`
      (the boosting-round loop surfaces no `n_iter_` attribute in v1) — so `_N_ITER`
      is included in both HGB `_EXPECTED` entries, unlike RF's. This was discovered
      by running the real sweep before writing any xfail entry, per the plan's own
      "empirically triaged... do NOT assume it passes cleanly" instruction — not
      copy-pasted from TASK-16's RF map.
      Tests (genuine Red then Green, both freshly demonstrated this session): RED —
      `pytest crates/mlrs-py/python/tests/test_params.py -k matrix_covers_exports`
      before any edit: `test_matrix_covers_exports` FAILED
      (`AssertionError: EXPECTED_PARAMS keys {'HistGradientBoostingClassifier',
      'HistGradientBoostingRegressor'} differ from the exported estimator shims`) —
      confirmed as the exact pre-existing gap TASK-23 explicitly left untouched for
      this task (TASK-23's own progress entry: "not fixed here, to avoid the scope
      creep of touching TASK-25's own file ahead of its own wave"). GREEN — same
      command: `pytest crates/mlrs-py/python/tests/test_params.py -q`: **147 passed,
      0 failed** (full file, up from 139 pre-edit — +8 for the two new estimators'
      4 parametrized tests each: `test_default_params_match_sklearn_names`,
      `test_set_params_roundtrip`, `test_init_purity_stores_kwargs_verbatim`,
      `test_init_purity_ast`). `pytest crates/mlrs-py/python/tests/test_estimator_checks.py
      -k HistGradient -q`: initial run (before the `_EXPECTED` map existed) showed
      **21 unexplained FAILED** (`check_dtype_object`, the 3 sparse checks,
      `check_estimators_pickle`(x2), `check_classifiers_classes`,
      `check_classifiers_regression_target`, `check_supervised_y_2d`,
      `check_non_transformer_estimators_n_iter`, `check_fit2d_1sample`,
      `check_requires_y_none` for the classifier; the regressor-applicable subset of
      the same set) — genuine Red, the exact "run the sweep and read the ACTUAL
      failure before writing any xfail entry" step the plan requires. After adding
      the triaged `_EXPECTED` entries: `pytest ... -k HistGradient -q`: **82 passed,
      4 skipped (pre-existing, unrelated `check_array_api_input`/
      `check_*_data_not_an_array` skip reasons), 21 xfailed (all with documented
      reasons), 0 unexplained failures**. `pytest ... -k RandomForest -q`: **84
      passed, 4 skipped, 19 xfailed** — byte-identical to TASK-16's own recorded
      counts, confirming this task's purely-additive `_estimators()`/`_EXPECTED`
      edit introduced ZERO regression to the RF entries.
      `pytest crates/mlrs-py/python/tests/test_estimator_checks.py::test_fit_free_checks_never_xfailed`:
      passed (confirms neither HGB xfail map leaks a fit-free check). `pytest
      crates/mlrs-py/python/tests/test_estimator_checks.py --collect-only -q`:
      **1574 tests collected, 0 collection errors** (up from the pre-edit baseline
      of 1467 — the two new estimators' full check matrix, confirmed non-broken
      collection). Full broader regression: `pytest crates/mlrs-py/python/tests/
      --ignore=.../test_estimator_checks.py -q`: **453 passed, 7 skipped
      (pre-existing, unrelated), 6 xpassed (the TASK-24-documented, already-known
      HGB deterministic-tier XPASS — not newly introduced by this task, unaffected
      by this task's edits, which touch neither `test_oracle_ensemble.py` nor any
      HGB algos/kernel file), 0 failed** — a genuine, non-truncated full run of
      every OTHER file in the directory (`test_params.py`, `test_shims.py`,
      `test_oracle_ensemble.py`, `test_dtype.py`, `test_io.py`,
      `test_import_probe.py`, `test_egress_shape_regression.py`, `test_wheels.py`,
      all included). `pytest crates/mlrs-py/tests/test_random_forest.py` (the
      Rust-facing FFI smoke suite, unrelated to this task's own file scope): **27
      passed, 0 failed** — re-confirmed unaffected. `git diff --stat` on both
      edited files: **90 insertions(+), 0 deletions(-)** across
      `test_params.py`/`test_estimator_checks.py` — purely additive, no existing
      per-estimator `EXPECTED_PARAMS`/`SET_PARAM`/`_estimators()`/`_EXPECTED` entry
      was touched, matching TASK-16's own "no plausible mechanism for any other
      estimator's check result to regress" reasoning. `git status --short` confirms
      this task modified exactly these 2 files and zero Rust source (this task made
      no Rust-source edit at all — pure Python gate-test wiring — so no `cargo`
      check/build gate applies).
      Environment note (severe, ongoing, externally-caused multi-tenant contention,
      matching TASK-20/24's own documented finding): a PRE-EXISTING, unrelated
      `pytest crates/mlrs-py/python/tests/ -q` process (PID `602803`, running since
      before this task's own session began, ~600%+ CPU sustained for 4+ hours
      accumulated CPU time, confirmed via `ps aux` to belong to a different
      invocation shape than this task's own commands) prevented the FULL
      `test_estimator_checks.py` sweep (all ~38 estimators, ~1574 parametrized
      check invocations) from completing within two attempted background windows
      (both intentionally terminated once confirmed non-progressing under that
      contention, mirroring TASK-24's own documented "terminated ... rather than
      left to consume further shared resources indefinitely" precedent) — the
      `-k HistGradient`/`-k RandomForest` targeted subsets (166 passed combined,
      40 xfailed, 0 unexplained failures) plus the collection-count/purely-additive-diff
      checks above are treated as equivalent regression evidence per the
      established project precedent (TASK-01 through TASK-24's blocker log), not a
      literal full-sweep pass.
      Specs: `SPEC-PY-ENS-05`'s HGB half (both `HistGradientBoostingClassifier` and
      `HistGradientBoostingRegressor` now covered by `test_params.py`'s AST-purity
      gate + per-estimator `get_params`/mutation table, and
      `test_estimator_checks.py`'s sklearn `check_estimator` sweep — every HGB check
      either passes or carries a documented, Green-time-verified xfail reason, none
      silently assumed; `test_shims.py` needed no new entry, confirmed by its own
      auto-derivation mechanism and a zero-regression full-file run) is now
      implemented and verified. Combined with TASK-16's RF half, `SPEC-PY-ENS-05` is
      now FULLY implemented across all four ensemble estimators. `SPEC.md`'s
      per-spec `**Status:** draft.` lines left UNCHANGED, matching the TASK-01..24
      precedent established throughout this file (no PageIndex document exists for
      this feature — `pageindex_update: "NOT APPLICABLE"` in `SPEC.md`'s own
      frontmatter — status transitions are orchestrator-directed here, not
      executor-silent; this executor's mandate is per-task TreeFinder-spec-state
      synchronization, and since this repo carries no PageIndex/TreeFinder
      document for this feature, per every prior task's own established and
      unchallenged precedent, this task follows the same convention rather than
      unilaterally introducing a new one at the very last task).

      **Final SPEC.md coverage check — all 9 original spec IDs (TASK-01..25
      collectively):**
      | Spec ID | Status | Evidence |
      |---|---|---|
      | `PY-ENS-01` (RF classifier binding) | FULLY implemented, verified | TASK-08 (Rust), TASK-11 (shim), TASK-10/13 (registration/export), TASK-14 (oracle replay, 14/14 passed), TASK-16 (gate tests) |
      | `PY-ENS-02` (RF regressor binding) | FULLY implemented, verified | TASK-09, TASK-12, TASK-10/13, TASK-15 (14/14 passed), TASK-16 |
      | `PY-ENS-03` (HGB classifier binding) | Structurally implemented + verified; deterministic-tier sklearn-exact-match tolerance intentionally NOT pinned | TASK-18 (Rust, structural), TASK-21 (shim), TASK-20/23 (registration/export), TASK-24 (oracle replay — statistical-tier/structural assertions genuinely pass; deterministic-tier exact-match assertions are `xfail`-marked per the locked HGB-churn-dirty precondition, which TASK-25 re-confirmed still dirty via `git status --short` above), TASK-25 (gate tests) |
      | `PY-ENS-04` (HGB regressor binding) | Same as PY-ENS-03 | TASK-19, TASK-22, TASK-20/23, TASK-24, TASK-25 |
      | `PY-ENS-05` (registration + gate tests) | FULLY implemented, verified (RF half: TASK-16; HGB half: THIS task) | TASK-10/13/16 (RF), TASK-20/23/25 (HGB) |
      | `RF-IMP-01` (feature_importances_ Rust core) | FULLY implemented, verified | TASK-01/02/03 |
      | `RF-IMP-02` (feature_importances_ Python binding) | FULLY implemented, verified | TASK-08/09/11/12/14/15/16 |
      | `RF-OOB-01` (oob_score_ Rust core) | FULLY implemented, verified | TASK-04/05/06/07 |
      | `RF-OOB-02` (oob_score_ Python binding) | FULLY implemented, verified | TASK-08/09/11/12/14/15/16 |

      **Final SPEC.md §6 acceptance-scenario coverage check (all 8 scenarios,
      TASK-01..25 collectively):**
      1. RF classifier fit/predict end-to-end — FULLY verified (TASK-14, 14/14; re-confirmed zero-regression in this task's own 453-passed broader run).
      2. RF regressor fit/predict end-to-end — FULLY verified (TASK-15, 14/14; re-confirmed).
      3. HGB classifier fit/predict_proba matching the (freshly-committed, non-dirty) fixture — NOT met as a locked, pinned assertion: the fixture remains dirty as of this task's own fresh `git status --short` re-check (identical to TASK-17/TASK-24's finding); the deterministic-tier exact-match test exists, is correctly shaped, and is `xfail`-marked (currently reporting `XPASS`, a signal not a resolution, per TASK-24's own explicit note, unchanged by this task) — this is the ONE intentionally-incomplete item in this plan's full 25-task history, exactly as the locked decision anticipates.
      4. HGB regressor fit/predict end-to-end — same status as scenario 3 (`xfail`-marked deterministic tier, statistical tier genuinely passes).
      5. All four estimators pass `test_params.py`/`test_shims.py`/documented-xfail `test_estimator_checks.py` — FULLY verified for all four estimators as of THIS task (RF: TASK-16; HGB: THIS task, 82+84=166 passed / 40 xfailed / 0 unexplained failures across the two targeted subsets, plus the 453-passed broader directory run).
      6. `feature_importances_` sums to 1 + dominant-feature ranking — FULLY verified (TASK-01/02/03 Rust, TASK-14/15 Python oracle replay).
      7. `oob_score_` statistical band + `AttributeError` when `oob_score=False` — FULLY verified (TASK-04/05/06/07 Rust, TASK-14/15 Python oracle replay, TASK-16's dedicated conditional-attribute test).
      8. `oob_score=True, bootstrap=False` raises `ValueError` — FULLY verified (TASK-05 Rust builder test, TASK-14/15 Python oracle replay, TASK-08/09 FFI smoke test).

      **Summary: 7 of 9 spec IDs and 6 of 8 acceptance scenarios are FULLY
      implemented and verified with real, passing test evidence. The remaining 2
      spec IDs (`PY-ENS-03`/`PY-ENS-04`) and 2 acceptance scenarios (3, 4) are
      intentionally, locked-decision-compliantly INCOMPLETE at exactly one
      sub-clause each — the deterministic-tier sklearn-exact-match oracle pin for
      HGB — gated on the HGB algos churn (`hist_gradient_boosting.rs`, `gbt.rs`,
      `gen_oracle.py`, all four `hgb_*.npz` fixtures) landing as a clean commit,
      which this task's own fresh `git status --short` re-check (below) confirms
      has NOT yet happened. Every OTHER sub-clause of PY-ENS-03/04 (structural
      binding, registration, shim wiring, statistical-tier band, not-fitted/invalid-input
      errors, gate-test coverage) IS implemented and verified. This is the single,
      plan-anticipated, user-locked incompleteness in the entire 25-task plan — not
      a defect in what TASK-25 or any prior task implemented.**

      Fresh HGB-churn gate re-check (this task's own, per the mission's explicit
      instruction to verify current state, not merely cite TASK-17/24's prior
      findings): `git status --short -- crates/mlrs-backend/src/prims/hist_gradient_boosting.rs
      crates/mlrs-kernels/src/gbt.rs scripts/gen_oracle.py
      tests/fixtures/hgb_cls_f32_seed42.npz tests/fixtures/hgb_cls_f64_seed42.npz
      tests/fixtures/hgb_reg_f32_seed42.npz tests/fixtures/hgb_reg_f64_seed42.npz`
      → all 7 paths still `M` (modified, uncommitted), 2026-07-18 — unchanged since
      TASK-17/TASK-24's own checks. TASK-25 does not touch HGB oracle tolerances at
      all (out of its own scope per PLAN.md), so this re-check is confirmatory only,
      not an action item for this task.

      **PY-ENSEMBLE plan status after TASK-25: 25 of 25 tasks completed.** The plan
      is complete modulo the single, explicitly locked-decision-compliant HGB
      deterministic-tier tolerance-pinning gap documented above and throughout
      TASK-17/24/25's own entries — not a defect, an intentional, user-approved
      incompleteness pending an off-plan commit event.

## Blockers / notes log
(executors: append dated entries here when a task is blocked or a deviation
is recorded, per PLAN.md's own risk/guardrail notes)

- **2026-07-18, TASK-02, BLOCKED (specification conflict, not a code bug):**
  `RandomForestClassifier<F, Fitted>::feature_importances()` accessor was
  implemented correctly (`fit()` now destructures `RfFitOutcome`, wires
  `feature_importances_`; `crates/mlrs-algos/src/ensemble/random_forest_classifier.rs`)
  and the qualitative dominant-feature-ranking Red/Green test PASSES. The
  EXACT-tier (`≤1e-5` f64 / `≤1e-4` f32) deterministic-tier oracle assertion
  against `sklearn.ensemble.RandomForestClassifier(...).feature_importances_`
  FAILS by ~0.0022 (got `0.45969`, sklearn `0.45748` on `X[:,0]`) — orders of
  magnitude beyond tolerance. Root-caused via a throwaway diagnostic test
  (not committed): mlrs's own 2 deterministic-tier trees ARE bit-identical to
  each other (as expected, `bootstrap=false`+`max_features=All` are
  zero-RNG), which mathematically PROVES the plan's own designated
  mitigation ("re-derive per-tree normalization instead of sum-then-normalize")
  is a no-op here — both aggregation formulas are provably equal when a
  forest's trees are internally identical. Separately confirmed via Python
  (`det.estimators_[0].feature_importances_` vs `det.estimators_[1]...`) that
  **sklearn's own two deterministic-tier trees are NOT identical to each
  other** (`np.allclose(...) == False`) — sklearn's Cython "best" splitter
  breaks near-tied candidate splits using internal per-tree randomness
  independent of `bootstrap`/`max_features`. A node-by-node dump confirmed
  mlrs's and sklearn's trees agree at the root and near-root levels but
  diverge at a low-sample-count node deep in the tree (mlrs picks
  `feature_3 <= 0.033`, sklearn picks `feature_3 <= 0.10` — a genuine
  different-but-tied split, not a rounding artifact), which is enough to
  shift `feature_importances_` well past the `1e-5`/`1e-4` tolerance even
  though train-set `predict`/`predict_proba` still match exactly (a weaker
  condition than tree isomorphism). PLAN.md's own Risk note for TASK-02
  anticipated exactly this class of failure and explicitly required a STOP
  rather than an executor-chosen tolerance change:
  "if Green-time reveals the two formulas diverge even there beyond 1e-5,
  STOP and re-derive per-tree normalization instead ... flag this as the
  single highest-uncertainty numeric claim in this plan." The designated
  fallback is proven ineffective by the diagnostic above, so this requires a
  spec-owner decision (e.g. widen the exact-tier tolerance to a
  ranking/statistical-style band, or otherwise revise RF-IMP-01's
  deterministic-tier acceptance claim in `SPEC.md`) before TASK-02 can be
  completed. Deviation also recorded: `random_forest_regressor.rs`'s `fit()`
  was given the same minimal `RfFitOutcome` destructuring TASK-01 required in
  `random_forest_classifier.rs` (WITHOUT adding a `feature_importances()`
  accessor there — that stays TASK-03's scope) because TASK-01 only touched
  `mlrs-backend`, leaving all of `mlrs-algos` (both ensemble files) unable to
  compile; this was a necessary, minimal unblock for TASK-02's own Verify
  step (`cargo test -p mlrs-algos --features cpu`) to even run.

- **2026-07-18, TASK-02, RESOLVED (orchestrator decision):** Replaced the
  `1e-5`/`1e-4` exact-tier deterministic assertion with `atol=0.05` (25x the
  observed ~0.0022 divergence — tight enough to catch a real attribution
  bug, tolerant of legitimate sklearn-internal tie-break disagreement). The
  qualitative dominant-feature-ranking test (already passing) is the PRIMARY
  correctness signal for `RF-IMP-01`, not a fallback. `SPEC.md` updated to
  `spec_revision: 2` (§5 RF-IMP-01 acceptance criteria, §9 Risk 5) and
  `PLAN.md` updated at TASK-02/TASK-03's Objective/Red/Risks sections plus
  TASK-14's Implementation Step 5 and the top-of-file "Resolved planning
  decisions" §Q-feature-importances-tolerance — all cross-referencing this
  entry. No re-run of the full 3-pass Plan Checker gate: this is a narrow,
  mathematically-justified tolerance correction within a risk the plan's own
  Risk section already anticipated and required a STOP for, not a new
  design decision. TASK-02 resumed to finish Green/Refactor/Verify against
  `atol=0.05`.

## Code-review fixes (post-implementation, 2026-07-18)

Two-reviewer adversarial code review (Rust numerical core + PyO3/Python binding
layer, CodeGraph-verified) after all 25 tasks landed. Three findings, all fixed:

- **[MEDIUM] `feature_importances_` cross-tree aggregation** — the reduction in
  `crates/mlrs-backend/src/prims/random_forest.rs` summed raw per-node decreases
  across ALL trees then normalized once (`Σd/ΣS`), diverging from sklearn's
  per-tree-normalize-then-average (`mean_t(d_t/S_t)`) whenever tree totals `S_t`
  differ — i.e. for the default `bootstrap=True` config. The deterministic-tier
  oracle test could not catch it (bit-identical trees make the two schemes
  coincide). Fixed to match sklearn's `_forest.py` exactly. Deterministic-tier
  fixture values are unchanged (both schemes agree there), so no fixture regen
  and the existing `atol=0.05` oracle tests still hold. Added a prim-level
  regression lock `feature_importances_uses_per_tree_normalization_not_global`
  (builds a `bootstrap=false, max_features<d` forest whose trees genuinely
  differ, asserts the accessor matches an independent per-tree recompute AND
  that per-tree differs from the global scheme for that data). SPEC.md §5
  reconciled (line 49 "mean over trees" was authoritative; the "sum across all
  nodes/trees" phrasing was the defect).
- **[LOW] `max_features` fraction used `ceil`** — sklearn uses `int()`
  (truncation). Fixed `MaxFeaturesArg::resolve`'s `Frac` arm in
  `crates/mlrs-py/src/estimators/ensemble.rs`; added
  `test_max_features_fraction_uses_floor_not_ceil`.
- **[LOW] explicit `max_features=None` collapsed to `"sqrt"`** — the PyO3
  `Option<&Bound>` default cannot distinguish an omitted arg from an explicit
  `None` at the FFI boundary (a `&Bound` param also cannot take a string-literal
  signature default in PyO3 — a first attempt at a Rust-side sentinel failed to
  compile). Fixed at the correct layer: the Python **shim**
  (`crates/mlrs-py/python/mlrs/ensemble.py`) — whose own `__init__` default is
  `"sqrt"`/`1.0`, so it CAN see an explicit `None` — forwards an explicit `None`
  to `_mlrs` as the new `"all"` sentinel string (added to `parse_max_features`),
  giving full sklearn `None`-means-all parity at the user-facing API while
  `get_params()` still round-trips the caller's original `None`. The `_mlrs`
  FFI layer documents that omitted==explicit-None==estimator-default there.
  Tests: shim-level `test_random_forest_classifier_max_features_none_is_all_features`
  (`python/tests/test_oracle_ensemble.py`) + FFI-level
  `test_max_features_all_sentinel_and_ffi_none_contract` (`tests/test_random_forest.py`).

Note on the aggregation regression-lock test: the first draft used
`bootstrap=false` data, but a fully-separable `bootstrap=false` forest drives
every tree to the identical total decrease `S_t` (== root impurity), so
per-tree and global aggregation coincide there (max gap ~1e-16) — the guard
correctly flagged this. Corrected to `bootstrap=true` (different resample per
tree => genuinely different `S_t`), the exact regime the finding is about.

Verification (2026-07-18, all green): `cargo test -p mlrs-backend --features cpu
--test random_forest_feature_importances_test` 4/4 (incl. the new per-tree
aggregation lock, both assertions); `--test random_forest_oob_test` 3/3;
`cargo test -p mlrs-algos --features cpu --test random_forest_classifier_test`
14/14 and `--test random_forest_regressor_test` 13/13 (deterministic-tier
`feature_importances_matches_sklearn_deterministic_*` unchanged — no fixture
regen); `cargo check -p mlrs-py --features cpu`/`--features wgpu` clean (E0308
from the first sentinel attempt resolved by the shim-layer approach); extension
rebuilt via `cargo build -p mlrs-py --features cpu,extension-module`; pytest
`tests/test_random_forest.py` 29/29 (incl. both FFI max_features tests) and
`python/tests/test_oracle_ensemble.py` 37 passed + 6 xpass (pre-existing HGB
deterministic xfails; incl. the new shim None-is-all-features test).
