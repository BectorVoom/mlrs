---
plan_document: TDD implementation plan
phase: metrics-surface
source_spec: .planning/plans/metrics-surface/SPEC.md
source_research: .planning/plans/RESEARCH-METRICS.md
source_evidence: .planning/plans/metrics-surface/SOURCES.md
generated_at: 2026-07-16
task_count: 23
---

# mlrs sklearn Metrics Surface — TDD Implementation Plan

Plans from `SPEC.md` (16 spec IDs: `METR-INFRA-01`, `METR-CLS-01..09`,
`METR-REG-01..03`, `METR-BIND-01`, `METR-SHIM-01`, `METR-ORACLE-01`). Every
task below cites verified evidence (CodeGraph / Read) — no invented path or
symbol. Unverifiable details are marked `[UNVERIFIED]` and resolved at Green
time, not guessed here.

## Resolved planning decisions (Planner-owned open questions)

- **Q6 (sklearn version to stamp):** `scikit-learn==1.9.0`. Evidence: this is
  the most recently pinned version in `scripts/gen_oracle.py`'s own regen
  docstrings (`scripts/gen_oracle.py:931,3696` — HDBSCAN/UMAP generators,
  `[VERIFIED: LOCAL]`). No sklearn is installed in this planning environment
  (`ModuleNotFoundError: No module named 'sklearn'` — `[VERIFIED: LOCAL]`), so
  the Planner cannot query a live version; `1.9.0` is the best available
  in-repo precedent and MUST be the venv pin used at fixture-generation time
  (TASK-02). If the regen operator's actual installed version differs, the
  generator docstring must be updated to match and fixtures regenerated —
  this is a runtime fact, not a planning-time guess.
- **Q2 (`mean_squared_error`):** MSE-only, no `squared=` parameter — confirmed
  by SPEC.md non-goals (`[VERIFIED: WEB sklearn mean_squared_error docs]`,
  cited in SPEC §2/§9).
- **Q5 (`log_loss` eps):** fixed `1e-15` in the Rust/PyO3 layer; the Python
  shim additionally accepts `eps='auto'` and maps it to `1e-15` before calling
  `_mlrs.log_loss` (SPEC §4/§5).
- **Q7 (ingress convention):** plain `Vec<i32>`/`Vec<f64>` PyO3 extraction, NOT
  the arrow capsule (SPEC §4, `[VERIFIED: LOCAL crates/mlrs-py/src/ingress.rs:112-118]`
  — arrow ingress is float-only, rejecting the integer label vectors this
  surface needs).
- **PrfOut Python crossing:** the codebase has **no existing precedent** for a
  PyO3 function returning a Python-side "float or list" polymorphic value
  (`[VERIFIED: CODEGRAPH — grep across crates/mlrs-py/src/ for IntoPyObject/
  PyObject-branching returns found none]`). Rather than invent an unverified
  conversion pattern, `average=None` (`PrfOut::PerClass`) is bound as a
  **separate PyO3 function** (`..._per_class`, returning `Vec<f64>`), mirroring
  the codebase's existing convention of splitting by return arity/dtype
  suffix (e.g. `predict_proba_f32`/`predict_proba_f64` in
  `crates/mlrs-py/src/estimators/neighbors.rs:257,270`). The Python shim
  dispatches on `average is None` to pick which `_mlrs` function to call.
- **Q10 (OvO + `sample_weight`):** PROBED at TASK-02 Green time against the
  pinned `scikit-learn==1.9.0` — call `sklearn.metrics.roc_auc_score(y_true,
  y_score, multi_class='ovo', average='macro', sample_weight=w)` before
  writing any fixture. **Branch A** (raises): do NOT generate a weighted-OvO
  fixture; the Rust `roc_auc_score_multiclass` OvO branch returns
  `Err(MetricError::WeightedOvoUnsupported)` whenever `sample_weight.is_some()`
  (TASK-10/TASK-21 add the corresponding error-gate Red tests). **Branch B**
  (does not raise): generate `ref_roc_auc_ovo_macro_sw`/`ref_roc_auc_ovo_weighted_sw`
  and TASK-10/TASK-21 add value-matching Red tests instead of the error gate.
  Both branches are specified explicitly in TASK-02/TASK-10/TASK-21 below (SPEC
  §2/§4/§9 Q10, Plan-Check Issue 2). `MetricError` now carries a
  `WeightedOvoUnsupported` variant (SPEC §4) alongside the five from TASK-01's
  original design.
- **Multioutput (2-D regression `y`) is a NON-GOAL** (SPEC §2, revised —
  downgrades an earlier draft assumption). `r2_score`/`mean_squared_error`/
  `mean_absolute_error` stay 1-D-only at BOTH the Rust and PyO3 layers (no
  `multioutput` parameter anywhere in the Rust contract — SPEC §4 already
  reflects this); the Python shim fails closed with `NotImplementedError` on a
  2-D `y_true`/`y_pred` or a non-default `multioutput=` kwarg, rather than
  silently `ravel()`-ing a 2-D array into a mathematically wrong 1-D result
  (TASK-16 adds the Red test for this).
- **`MetricError` → `PyValueError` mapping** uses a NEW `metric_err_to_py`
  function (TASK-15), a SIBLING of the existing `algo_err_to_py`
  (`crates/mlrs-py/src/errors.rs:56-58`, `[VERIFIED: CODEGRAPH]`) — `AlgoError`
  and `MetricError` are distinct types (SPEC §4 explicit correction), so
  `algo_err_to_py` (which only accepts `AlgoError`) cannot be reused directly
  for `MetricError`.

## Fixture naming scheme (binds TASK-02 to TASK-03..14, TASK-17..23)

All fixtures live in `<repo>/tests/fixtures/` (`_FIXTURE_DIR`,
`scripts/gen_oracle.py:41`, `[VERIFIED: LOCAL]`), written via `np.savez`
(scalars as length-1 arrays, the established convention). **Every array in
every one of these files MUST be cast to `float32`/`float64` before
`np.savez` — including every label array (`y_true*`, `y_pred*`), every
`labels*` array, and every `ref_confusion*` count matrix** — because
`mlrs_core::oracle::load_npz` (`crates/mlrs-core/src/oracle.rs:115-135`,
`[VERIFIED: CODEGRAPH]`) only decodes 4-byte or 8-byte FLOAT dtypes per array
(`arr.dtype().num_bytes()` matched against `Some(8)`/`Some(4)` only) and
returns a hard `io::Error` — failing the ENTIRE fixture's load, not just the
offending array — the moment it meets an `int64`/`int32` array. TASK-02
below states this as an explicit Green-step requirement and a completion
criterion. Rust/Python tests read labels back via `expect_f64` and cast to
`i32`/`i64` on their own side (comparing confusion counts as `f64 == integer`
value, e.g. `(got - 6.0).abs() < 1e-9`), exactly as `fixture_vec::<F>` already
does for the RF/HGB float-labeled fixtures (`random_forest_classifier_test.rs:75-80`).

Filenames:

| Fixture file | Dtype suffix | Contents (named arrays) |
|---|---|---|
| `metrics_cls_binary_{f32,f64}_seed42.npz` | both | `y_true`,`y_pred` (float-cast {0,1} labels), `y_score` (positive-class score), `sample_weight`; `ref_accuracy`,`ref_accuracy_sw`; `ref_confusion`,`ref_confusion_sw`; `ref_precision_binary`,`ref_recall_binary`,`ref_f1_binary`; `ref_precision_binary_sw`,`ref_recall_binary_sw`,`ref_f1_binary_sw`; `ref_roc_auc`,`ref_roc_auc_sw`; `ref_pr_precision`,`ref_pr_recall`,`ref_pr_thresholds`; **`ref_pr_precision_sw`,`ref_pr_recall_sw`,`ref_pr_thresholds_sw`** (weighted PR-curve — Issue 1); `ref_log_loss_binary` (Issue 7 — explicit binary log_loss reference; needs a companion `y_prob_binary` 2-column row-major proba array) |
| `metrics_cls_multiclass_{f32,f64}_seed42.npz` | both | `y_true`,`y_pred` (float-cast {0,1,2} labels), `y_proba` (n×3 row-major), `sample_weight`; `ref_accuracy`,`ref_accuracy_sw`; `ref_confusion`; `ref_precision_{macro,micro,weighted,none}`, `ref_recall_{macro,micro,weighted,none}`, `ref_f1_{macro,micro,weighted,none}`; `ref_precision_macro_sw`,`ref_recall_macro_sw`,`ref_f1_macro_sw`; `ref_log_loss`,`ref_log_loss_sw`; `ref_roc_auc_ovr_macro`,`ref_roc_auc_ovr_weighted`,`ref_roc_auc_ovo_macro`,`ref_roc_auc_ovo_weighted`; **`ref_roc_auc_ovr_macro_sw`,`ref_roc_auc_ovr_weighted_sw`** (weighted OvR — Issue 1, ALWAYS generated); **`ref_roc_auc_ovo_macro_sw`,`ref_roc_auc_ovo_weighted_sw`** (weighted OvO — Issue 2, generated ONLY if the TASK-02 probe shows the pinned sklearn accepts `sample_weight` with `multi_class='ovo'`; otherwise ABSENT and the Rust/Python tests assert the error gate instead); `y_true_labelreorder`,`y_pred_labelreorder`,`labels_reorder` (a permuted `labels=[2,0,1]`-style order) with `ref_precision_labelreorder`,`ref_recall_labelreorder`,`ref_f1_labelreorder` (Issue 6, precision/recall/f1 `labels`-reorder acceptance) |
| `metrics_cls_degenerate_seed42.npz` | f64 only, EVERY array float-cast (exact/int-VALUED cases, not int-DTYPED — see the float-cast note above) | `y_true_empty`,`y_pred_empty`,`labels_empty`(=[0,1,2]),`ref_confusion_empty`(3×3, zero row/col for class 2); `y_true_one`,`y_pred_one`,`ref_confusion_one`([[n]]); `y_true_zp`,`y_pred_zp`,`ref_precision_zerodiv`; `y_true_zr`,`y_pred_zr`,`ref_recall_zerodiv`; `y_true_zf`,`y_pred_zf`,`ref_f1_zerodiv`; `y_true_single_match`,`y_pred_single_match`,`ref_acc_single_match`(=1.0); `y_true_single_mismatch`,`y_pred_single_mismatch`,`ref_acc_single_mismatch`(=0.0); `y_true_singleclass`,`y_score_singleclass` (single-class roc_auc — NO ref value, an error gate); `y_true_clip`,`y_prob_clip`,`ref_log_loss_clip`; `y_true_logloss_labelreorder`,`y_prob_logloss_labelreorder`,`labels_logloss_reorder`,`ref_log_loss_labelreorder` (Issue 6, log_loss `labels`-reorder acceptance, binary so it is cheap alongside Issue 7's binary log_loss case) |
| `metrics_reg_{f32,f64}_seed42.npz` | both | `y_true`,`y_pred`,`sample_weight` (1-D, single-output ONLY — no 2-D array in this fixture, per SPEC §2's multioutput non-goal); `ref_r2`,`ref_r2_sw`,`ref_mse`,`ref_mse_sw`,`ref_mae`,`ref_mae_sw`; `y_true_const`,`y_pred_const`,`ref_r2_const`; `y_perfect` (reused as both true/pred), `ref_r2_perfect`(=1.0),`ref_mse_perfect`(=0.0),`ref_mae_perfect`(=0.0) |

`_sw` = the `sample_weight`-applied variant of the same metric (non-uniform
weights), satisfying the "sample_weight on every metric" cross-cutting
requirement without an exhaustive per-average × weighted cross-product,
EXCEPT the documented OvO carve-out (Q10 above) where `_sw` may be entirely
absent by design.

## Execution waves (dependency order)

```text
Wave 1: TASK-01 (METR-INFRA-01)
Wave 2: TASK-02 (METR-ORACLE-01)              [depends on: none; parallel with TASK-01]
Wave 3a (classification, sequential, same file):
  TASK-03 -> TASK-04 -> TASK-05 -> TASK-06 -> TASK-07 -> TASK-08 -> TASK-09 -> TASK-10 -> TASK-11
Wave 3b (regression, sequential, same file; PARALLEL with Wave 3a — disjoint files):
  TASK-12 -> TASK-13 -> TASK-14
Wave 4: TASK-15 (METR-BIND-01)                 [depends on: TASK-03..14 all landed]
Wave 5: TASK-16 (METR-SHIM-01)                 [depends on: TASK-15]
Wave 6 (Python oracle replay, sequential, same file; depends on TASK-16 + TASK-02):
  TASK-17 -> TASK-18 -> TASK-19 -> TASK-20 -> TASK-21 -> TASK-22 -> TASK-23
```

Wave 1 and Wave 2 have no mutual dependency (different files: `metrics/mod.rs`
+ `lib.rs` vs. `scripts/gen_oracle.py` + fixtures) — parallel-eligible.

**Revised parallelism fix (Plan-Check Issue 4):** TASK-01 now CREATES both
`crates/mlrs-algos/src/metrics/classification.rs` and
`crates/mlrs-algos/src/metrics/regression.rs` as (near-)empty stub files AND
adds BOTH `pub mod classification;` and `pub mod regression;` to
`metrics/mod.rs` in its own Green step — `metrics/mod.rs` is edited EXACTLY
ONCE in the whole plan, entirely within TASK-01. TASK-03 (first classification
task) and TASK-12 (first regression task) therefore only APPEND functions
into their own already-`pub mod`-registered, already-existing file; neither
touches `metrics/mod.rs` again. This makes the Wave 3a ∥ Wave 3b parallelism
claim below actually valid (previously both waves' first task edited
`metrics/mod.rs`, a write conflict). Wave 3a and Wave 3b touch disjoint files
(`classification.rs`/`metrics_classification_test.rs` vs.
`regression.rs`/`metrics_regression_test.rs`) and both only need Wave 1's
shared types (already present in `metrics/mod.rs` after TASK-01) — genuinely
parallel-eligible with each other, but each wave is internally sequential
(same file per task, later task's Red step appends to the prior task's Green
file). TASK-15 needs every algos function to exist, so it waits for the LAST
task in both 3a and 3b. TASK-17..23 all edit
`crates/mlrs-py/python/tests/test_oracle_metrics.py` — sequential.

---

## TASK-01 — METR-INFRA-01: host metrics module scaffolding + shared bookkeeping

- **Spec:** `METR-INFRA-01`
- **Order:** 1 (Wave 1)
- **Depends on:** none
- **Parallel with:** TASK-02

### Objective
After this task, `crates/mlrs-algos/src/metrics/mod.rs` exists with the shared
`Average`, `ZeroDivision`, `MultiClass`, `MetricError`, `PrfOut` types and a
shared label/weight bookkeeping function (unique-class discovery + per-class
weighted TP/FP/FN accumulation), registered via `pub mod metrics;` in
`crates/mlrs-algos/src/lib.rs`. **This task ALSO creates the (near-)empty
`crates/mlrs-algos/src/metrics/classification.rs` and
`crates/mlrs-algos/src/metrics/regression.rs` stub files and wires BOTH
`pub mod classification;` and `pub mod regression;` into `metrics/mod.rs` up
front (Plan-Check Issue 4)** — this is the ONLY task in the whole plan that
edits `metrics/mod.rs`; every later classification/regression task
(TASK-03..14) only appends functions to its own already-registered,
already-existing file. No metric VALUE logic lives here yet — only the
substrate every classification/regression metric (TASK-03..14) builds on.

### Context and Evidence
- `crates/mlrs-algos/src/lib.rs:49-68` — module list pattern (`pub mod cluster;` … `pub mod projection;`), single `pub use error::AlgoError;` re-export at crate root (`[VERIFIED: CODEGRAPH]`). New module goes in the same flat list.
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs:156-181` — `accuracy_score` precedent: plain host `fn`, `assert_eq!` length guard, no CubeCL (`[VERIFIED: CODEGRAPH]`). The new bookkeeping follows the same plain-Rust, no-device style.
- SPEC.md §4 (revised) — exact enum/error shapes: `Average{Binary,Macro,Micro,Weighted,None_}`, `ZeroDivision{Zero,One,Nan}`, `MultiClass{Ovr,Ovo}`, `MetricError{LengthMismatch,EmptyInput,SingleClassRocAuc,BadShape,InvalidWeight,WeightedOvoUnsupported}` (the `WeightedOvoUnsupported` variant is NEW — SPEC §4 revision, Plan-Check Issue 2 — reserved here for TASK-10's OvO carve-out even though nothing constructs it until TASK-10), `PrfOut{Scalar(f64),PerClass(Vec<f64>)}`.
- No CubeCL/BufferPool/Pod bounds needed — host-only (SPEC §3, RESEARCH-METRICS §2-3).
- SPEC §4 also renames the error-mapping function this type feeds: `metric_err_to_py` (a NEW sibling of `algo_err_to_py`), owned by TASK-15, not this task — `MetricError` itself has no PyO3-facing code here.

### Files
- Create: `crates/mlrs-algos/src/metrics/mod.rs`
- Create: `crates/mlrs-algos/src/metrics/classification.rs` (near-empty stub: a module doc-comment only, no functions — TASK-03 appends the first function)
- Create: `crates/mlrs-algos/src/metrics/regression.rs` (near-empty stub: a module doc-comment only, no functions — TASK-12 appends the first function)
- Create: `crates/mlrs-algos/tests/metrics_infra_test.rs`
- Modify: `crates/mlrs-algos/src/lib.rs` (add `pub mod metrics;` to the flat module list, alphabetically after `manifold` and before `naive_bayes` per the existing alphabetical ordering at lines 49-63)

### TDD Sequence

#### 1. Red
- Test name: `unique_classes_and_weighted_counts_from_labels` in `crates/mlrs-algos/tests/metrics_infra_test.rs`.
- Setup: hand-built `y_true = [0,1,0,2]`, `y_pred = [0,1,1,2]`, no `sample_weight`, no explicit `labels`.
- Call the not-yet-existing `mlrs_algos::metrics::class_bookkeeping(&y_true, &y_pred, None, None) -> Result<ClassBookkeeping, MetricError>` (or equivalent shared helper — exact fn name decided at Green, but must be `pub` under `mlrs_algos::metrics`).
- Expected: sorted unique classes `[0,1,2]`; per-class weighted TP/FP/FN equal to their unweighted integer counts (class 0: TP=1,FP=0,FN=1; class1: TP=1,FP=1,FN=0; class2: TP=1,FP=0,FN=0).
- Expected initial failure: compile error — `mlrs_algos::metrics` module does not exist yet.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_infra_test unique_classes_and_weighted_counts_from_labels`

#### 2. Green
- Create `crates/mlrs-algos/src/metrics/mod.rs` with:
  - `pub enum Average { Binary, Macro, Micro, Weighted, None_ }`
  - `pub enum ZeroDivision { Zero, One, Nan }`
  - `pub enum MultiClass { Ovr, Ovo }`
  - `pub enum MetricError { LengthMismatch, EmptyInput, SingleClassRocAuc, BadShape, InvalidWeight, WeightedOvoUnsupported }` (+ `impl std::fmt::Display`/`std::error::Error` mirroring `AlgoError`'s style so TASK-15's NEW `metric_err_to_py` mapping is trivial later — `WeightedOvoUnsupported` is unused until TASK-10 but must exist now so `mod.rs` is edited exactly once in the whole plan)
  - `pub enum PrfOut { Scalar(f64), PerClass(Vec<f64>) }`
  - a bookkeeping function returning sorted unique classes (or the caller's `labels` verbatim, including labels absent from the data) and per-class weighted `(tp, fp, fn)` `f64` triples, validating `y_true.len() == y_pred.len()` and (if given) `sample_weight.len() == y_true.len()` and every weight is finite and `>= 0.0`, returning `Err(MetricError::LengthMismatch)` / `Err(MetricError::InvalidWeight)` on violation (no panic — SPEC §5 behavior clause).
  - `pub mod classification;` and `pub mod regression;` (both added here, Plan-Check Issue 4 — the ONLY point in the plan `metrics/mod.rs` is touched).
- Create `crates/mlrs-algos/src/metrics/classification.rs` as a stub: a module doc-comment stating "classification metrics land here starting TASK-03 (METR-CLS-01)" and nothing else (no `fn`s yet — an empty file with only a doc-comment compiles cleanly as a `pub mod`).
- Create `crates/mlrs-algos/src/metrics/regression.rs` as a stub: same pattern, "regression metrics land here starting TASK-12 (METR-REG-01)".
- Add `pub mod metrics;` to `crates/mlrs-algos/src/lib.rs`.
- Do NOT implement `accuracy_score`/`confusion_matrix`/etc. here (TASK-03 owns that).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_infra_test unique_classes_and_weighted_counts_from_labels`

#### 3. Refactor
- Ensure the bookkeeping function's return type and field names are the ones TASK-03..11 will consume directly (no dead re-shaping later) — check against SPEC §4's per-metric signatures before finalizing names.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_infra_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_infra_test`
- Run: `cargo build -p mlrs-algos --features cpu` (confirm `pub mod metrics;` compiles cleanly with no other crate touched)
- Confirm: the four shared types + bookkeeping function are `pub` and importable as `mlrs_algos::metrics::{Average, ZeroDivision, MultiClass, MetricError, PrfOut}`.

### Implementation Steps
1. Add a second Red test: `length_mismatch_and_bad_weight_return_typed_errors` — `y_true.len() != y_pred.len()` → `Err(MetricError::LengthMismatch)`; a negative or NaN `sample_weight` entry → `Err(MetricError::InvalidWeight)` (no panic in either case — SPEC §5 explicit no-panic requirement).
2. Add a third Red test: `explicit_labels_include_absent_class_with_zero_counts` — `labels=[0,1,2]` where class `2` never appears in `y_true`/`y_pred`; bookkeeping still reports class `2` with `(tp,fp,fn)=(0,0,0)`.
3. Implement `mod.rs` to satisfy all three tests together (they share one Green pass since they exercise one function).
4. Wire `pub mod metrics;` into `lib.rs`.

### Completion Criteria
- [x] All three Red tests fail for the stated reason before `mod.rs` exists.
- [x] All three tests pass after `mod.rs` lands.
- [x] `cargo build -p mlrs-algos --features cpu` is clean.
- [x] No metric VALUE function (accuracy/confusion/etc.) is implemented here.
- [x] `crates/mlrs-algos/src/metrics/classification.rs` and `regression.rs` both exist as stub files, and `metrics/mod.rs` already contains `pub mod classification;` + `pub mod regression;` before TASK-03/TASK-12 start (Plan-Check Issue 4 — verified by grepping `metrics/mod.rs` for both lines at the end of this task).
- [x] `MetricError` includes `WeightedOvoUnsupported` (unused until TASK-10, but present now).

### Risks and Guardrails
- Risk: over-designing the bookkeeping API before TASK-03 discovers its exact needed shape. Mitigation: keep the bookkeeping function's output a simple `Vec<(i32 /*class*/, f64 /*tp*/, f64 /*fp*/, f64 /*fn*/)>` plus the resolved `Vec<i32>` class order — the smallest shape that satisfies precision/recall/f1's shared need (SPEC §5 CLS-03/04/05 note: "computed from the same weighted TP/FP/FN").
- Risk: a later task re-touching `metrics/mod.rs` (e.g. to add a helper re-export) would silently re-introduce the Wave-3a/3b write conflict Issue 4 fixes. Mitigation: TASK-03..14's Files lists explicitly state they do NOT modify `metrics/mod.rs` (verified below in each task) — any future need to add something to `mod.rs` must be a NEW task in a later wave, not folded into a 3a/3b task.

---

## TASK-02 — METR-ORACLE-01: oracle generators + committed fixtures

- **Spec:** `METR-ORACLE-01`
- **Order:** 1 (Wave 2)
- **Depends on:** none
- **Parallel with:** TASK-01

### Objective
After this task, `scripts/gen_oracle.py` has `gen_metrics_classification_binary`,
`gen_metrics_classification_multiclass`, `gen_metrics_classification_degenerate`,
`gen_metrics_regression` generators (docstring-stamped `scikit-learn==1.9.0`
per the Q6 resolution above), `main()` calls all four, and the fixtures listed
in the (revised) naming-scheme table above — INCLUDING the weighted
`precision_recall_curve` arrays, the weighted OvR `roc_auc_score` arrays, the
probe-gated weighted OvO arrays, the `labels`-reorder arrays for
precision/recall/f1 and log_loss, and the explicit binary `log_loss`
reference — are generated with EVERY array float-cast and **committed** to
`tests/fixtures/`. No Rust/Python test in this plan can pass without these
fixtures existing first — this task is the hard prerequisite for TASK-03..14
and TASK-17..23's Red steps to fail for the *right* reason (missing
implementation, not missing fixture).

### Context and Evidence
- `scripts/gen_oracle.py:41` — `_FIXTURE_DIR` (`[VERIFIED: LOCAL]`).
- `scripts/gen_oracle.py:3453-3519` (`gen_hgb_regressor`) and `:3522-3603` (`gen_hgb_classifier`) — the `np.savez(out_path, ...)` + `c(arr)` dtype-cast pattern to mirror exactly (`[VERIFIED: CODEGRAPH]`).
- `scripts/gen_oracle.py:3606-3856` (`main()`) — dispatch pattern: one `print(f"wrote {gen_x(dtype=dtype)}")` per (generator, dtype) pair, grouped under a `# ---- Phase-N ... ----` comment banner (`[VERIFIED: CODEGRAPH]`). New calls append after the `gen_hgb_classifier` block (the last currently in `main()`, ending at line 3852) and before `if __name__ == "__main__":` (line 3855).
- `scripts/gen_oracle.py:15-17` — regen venv instructions (`python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn`) (`[VERIFIED: LOCAL]`).
- `scripts/gen_oracle.py:931,3696` — most recent in-repo sklearn version pin (`scikit-learn==1.9.0`) — the Q6 resolution basis (`[VERIFIED: LOCAL]`).
- `crates/mlrs-core/src/oracle.rs:115-135` (`read_named_arrays`) — `[VERIFIED: CODEGRAPH]`: `num_bytes` is matched ONLY against `Some(8)` (f64) / `Some(4)` (f32); any other byte width (e.g. an `int64`/`int32` array saved without a float cast) hits the `other => Err(...)` arm and fails the ENTIRE `load_npz` call for that file — a single un-cast label array poisons every array in the same fixture. This is the Plan-Check Issue 5 root cause; the Green step below states the float-cast rule as a hard requirement, not a style preference.
- SPEC §2/§4/§9 Q10 (revised) — the OvO + `sample_weight` probe requirement (Plan-Check Issue 2), executed in this task's Green step BEFORE any weighted-OvO fixture decision is made.

### Files
- Modify: `scripts/gen_oracle.py`
- Create (committed data, not source): `tests/fixtures/metrics_cls_binary_f32_seed42.npz`, `tests/fixtures/metrics_cls_binary_f64_seed42.npz`, `tests/fixtures/metrics_cls_multiclass_f32_seed42.npz`, `tests/fixtures/metrics_cls_multiclass_f64_seed42.npz`, `tests/fixtures/metrics_cls_degenerate_seed42.npz`, `tests/fixtures/metrics_reg_f32_seed42.npz`, `tests/fixtures/metrics_reg_f64_seed42.npz`
- Create: `crates/mlrs-algos/tests/metrics_fixtures_exist_test.rs` (the Red/Green smoke test for this task)

### TDD Sequence

#### 1. Red
- Test name: `metrics_cls_binary_fixture_has_expected_arrays` in `crates/mlrs-algos/tests/metrics_fixtures_exist_test.rs`.
- Setup: `mlrs_core::load_npz(<workspace_root>/tests/fixtures/metrics_cls_binary_f64_seed42.npz)` (path helper mirrors `fixture()` in `crates/mlrs-algos/tests/random_forest_classifier_test.rs:50-57`, `[VERIFIED: CODEGRAPH]`).
- Expected: `load_npz` returns `Err` (file does not exist) — asserted via `.is_err()` or, if choosing to assert presence via `Result::expect` panicking, the test fails with "No such file or directory".
- Expected initial failure: the fixture file is absent (`std::io::Error` — `NotFound`), NOT an assertion-content mismatch. This is the correct Red state per the parent instruction "fixtures must exist before the tests that load them."
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_fixtures_exist_test metrics_cls_binary_fixture_has_expected_arrays`

#### 2. Green
- **Float-cast rule (Plan-Check Issue 5 — apply to EVERY array below, no exceptions):** every array passed to `np.savez`, including every `y_true*`/`y_pred*` label array, every `labels*` array, and every `ref_confusion*` count matrix, MUST go through the same `c(arr) = np.ascontiguousarray(np.asarray(arr)).astype(dtype)` cast used for the numeric reference arrays (`dtype` = the fixture's own `f32`/`f64` for the dtype-suffixed files; `np.float64` for `metrics_cls_degenerate_seed42.npz`, which has no dtype axis). Label/count arrays that are conceptually integers are stored as float-valued floats (e.g. `3.0`, not the int64 `3`) — `mlrs_core::oracle::load_npz` (`crates/mlrs-core/src/oracle.rs:115-135`) only decodes 4-/8-byte FLOAT dtypes and fails the WHOLE file otherwise. Do not save any `np.int32`/`np.int64`-dtyped array into any `metrics_*.npz` file.
- **OvO + `sample_weight` probe (Plan-Check Issue 2 / Q10) — run BEFORE writing `gen_metrics_classification_multiclass`'s weighted-OvO arrays:** in the regen venv, execute
  ```python
  import numpy as np
  from sklearn.metrics import roc_auc_score
  try:
      roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro", sample_weight=w)
      OVO_WEIGHT_SUPPORTED = True
  except (ValueError, TypeError) as exc:
      OVO_WEIGHT_SUPPORTED = False
  ```
  on the SAME `y_true`/`y_proba`/`sample_weight` the multiclass generator builds. **Branch A (`OVO_WEIGHT_SUPPORTED=False`):** do NOT compute or save `ref_roc_auc_ovo_macro_sw`/`ref_roc_auc_ovo_weighted_sw` — TASK-10/TASK-21 test the `Err`/`ValueError` gate instead. **Branch B (`True`):** compute and save both, and TASK-10/TASK-21 test the values. Record which branch fired in the generator's docstring (e.g. `"""... OvO+sample_weight probed under scikit-learn==1.9.0: {raises / does not raise} — see PLAN.md TASK-02 Q10."""`) so later readers do not have to re-probe.
- Implement the four generators in `scripts/gen_oracle.py`:
  - `gen_metrics_classification_binary(seed=SEED, dtype=np.float32) -> str` — builds a small binary `y_true`/`y_pred` set + `y_score` (positive-class continuous scores, tie-heavy to exercise rank ties) + non-uniform `sample_weight` + a `y_prob_binary` 2-column row-major proba array; computes every `ref_*` array in the naming-scheme table via `sklearn.metrics.{accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, log_loss}` — INCLUDING the weighted `precision_recall_curve` triple (`ref_pr_precision_sw`/`ref_pr_recall_sw`/`ref_pr_thresholds_sw`, called with `sample_weight=sample_weight` — Issue 1, now mandatory, not optional) and the explicit `ref_log_loss_binary` (Issue 7).
  - `gen_metrics_classification_multiclass(seed=SEED, dtype=np.float32) -> str` — 3-class `y_true`/`y_pred` + `y_proba` (row-major, rows summing to 1) + `sample_weight`; every `ref_*` per the table via the same sklearn functions with `average` swept over `{macro,micro,weighted,None}` plus `log_loss` and `roc_auc_score(multi_class={'ovr','ovo'}, average={'macro','weighted'})` — INCLUDING the weighted OvR pair `ref_roc_auc_ovr_macro_sw`/`ref_roc_auc_ovr_weighted_sw` (Issue 1, ALWAYS generated — OvR has no carve-out) and the probe-gated weighted OvO pair (see the probe step above); plus a `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` triple (a `labels` permutation, e.g. `[2,0,1]`) with `ref_precision_labelreorder`/`ref_recall_labelreorder`/`ref_f1_labelreorder` computed via `sklearn.metrics.{precision,recall,f1}_score(..., labels=[2,0,1], average='macro')` (Issue 6).
  - `gen_metrics_classification_degenerate(seed=SEED) -> str` — the hand-built degenerate cases from the naming-scheme table (empty-class confusion, all-one-class confusion, zero-division precision/recall/f1, single-sample accuracy match/mismatch, single-class roc_auc inputs — no ref value for the last, it is an error gate — log_loss clipping) PLUS a binary `y_true_logloss_labelreorder`/`y_prob_logloss_labelreorder`/`labels_logloss_reorder` triple with `ref_log_loss_labelreorder` via `sklearn.metrics.log_loss(..., labels=[1,0])` (Issue 6). f64 only, every array float-cast per the rule above (SPEC §6: exact/integer-VALUED-tier cases; no f32 variant needed since these are hand-built tiny arrays, not statistical).
  - `gen_metrics_regression(seed=SEED, dtype=np.float32) -> str` — continuous 1-D `y_true`/`y_pred` + `sample_weight` (single-output only — no 2-D array anywhere in this generator, per SPEC §2's multioutput non-goal); `ref_r2`/`ref_mse`/`ref_mae` (+ `_sw` weighted variants) via `sklearn.metrics.{r2_score, mean_squared_error, mean_absolute_error}`; plus the constant-target (`y_true_const` all-equal) and perfect-prediction (`y_perfect`) degenerate pairs, computing `ref_r2_const` from the ACTUAL sklearn output (do not hand-derive — SPEC §5 REG note) and `ref_r2_perfect=1.0`/`ref_mse_perfect=0.0`/`ref_mae_perfect=0.0`.
  - Every generator docstring opens with: `"""...Requires ``scikit-learn==1.9.0`` (the version this fixture was pinned against — see the Q6 resolution in .planning/plans/metrics-surface/PLAN.md)."""` mirroring the existing per-generator regen-venv comment convention (`gen_oracle.py:3696`).
  - Add the eight `main()` dispatch lines (binary ×2 dtypes, multiclass ×2 dtypes, degenerate ×1, regression ×2 dtypes) after the existing `gen_hgb_classifier` block, under a new `# ---- Metrics surface fixtures (METR-01/02/03) ----` banner.
- Regen: `python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn==1.9.0 && /tmp/oracle-venv/bin/python scripts/gen_oracle.py` — commit the seven new `.npz` files under `tests/fixtures/`.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_fixtures_exist_test metrics_cls_binary_fixture_has_expected_arrays`

#### 3. Refactor
- Factor the repeated `def c(arr): return np.ascontiguousarray(np.asarray(arr)).astype(dtype)` + `dtype_tag` + `os.makedirs` boilerplate into a shared local helper ONLY if it does not diverge from the existing per-generator inline convention (every existing generator repeats this inline — `[VERIFIED: CODEGRAPH gen_kernel_density:2496-2499, gen_hgb_regressor:3503-3506]`); prefer matching the established repetition over introducing a new abstraction the rest of the file does not use, to keep the diff minimal and reviewable.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_fixtures_exist_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_fixtures_exist_test`
- Confirm: `ls tests/fixtures/metrics_*.npz` lists exactly the 7 committed files; `git status` shows them as new/tracked.
- Confirm: every array name in the naming-scheme table is present (extend the Red test into a full name-presence assertion using `OracleCase::names()`, `crates/mlrs-core/src/oracle.rs:35-37`, `[VERIFIED: CODEGRAPH]`).
- Confirm: `load_npz` on each of the 7 files returns `Ok` and `OracleCase::f64(name)` is `Some(..)` for EVERY array name in the naming-scheme table (a `None`/decode error here means an un-cast integer array slipped through — re-check the float-cast rule).

### Implementation Steps
1. Write the Red smoke test asserting fixture presence + array-name completeness for all 7 files (7 sub-assertions in one test file, one `#[test]` per fixture file).
2. Run the OvO + `sample_weight` probe against the pinned sklearn BEFORE writing the multiclass generator's weighted-OvO branch; record the outcome in the generator's docstring.
3. Implement the 4 generators (with the float-cast rule applied to every array) + `main()` wiring.
4. Regenerate in a fresh `/tmp/oracle-venv` (PEP 668) pinned to `scikit-learn==1.9.0`.
5. Commit the fixtures (this plan does not run `git commit`; the operator commits per the user's own workflow — this task's Green is "fixtures exist on disk and are staged").

### Completion Criteria
- [x] Red test fails with `NotFound` / missing-array errors before generation.
- [x] All 7 fixtures exist with every named array from the (revised) table, including the weighted `pr_curve`, weighted OvR `roc_auc`, probe-gated weighted OvO `roc_auc`, `labels`-reorder (P/R/F1 + log_loss), and binary `log_loss` arrays.
- [x] `main()` calls all 4 new generators.
- [x] Every generator docstring stamps `scikit-learn==1.9.0`.
- [x] **No integer-dtype array is saved into any `metrics_*.npz` file** (Plan-Check Issue 5) — every array, including every label/`labels`/confusion-count array, is `float32`/`float64`-cast.
- [x] The OvO + `sample_weight` probe ran and its outcome (raises / does not raise) is recorded in the multiclass generator's docstring, determining whether `ref_roc_auc_ovo_*_sw` exists. **Outcome: RAISES (Branch A)** — confirmed against the installed `scikit-learn==1.9.0`.

### Risks and Guardrails
- Risk (flagged, not resolved here): the constant-target `r2_score` value and the exact `log_loss` clipping/renormalization behavior are `[UNVERIFIED]` against the Planner's memory of sklearn internals — SPEC §5 REG note explicitly requires pinning "sklearn's actual value," so this task's Green step MUST read the value the installed `scikit-learn==1.9.0` actually produces and store it verbatim; TASK-08/TASK-12 assert against the STORED fixture value, never a hand-derived one.
- Risk: single-class `roc_auc_score` input must NOT carry a `ref_roc_auc_singleclass` array (calling `sklearn.metrics.roc_auc_score` on it raises `ValueError` at generation time too) — the generator must NOT call it; only store the raw `y_true_singleclass`/`y_score_singleclass` inputs for the Rust/Python test to independently assert the `ValueError`/`Err` gate.
- Risk: forgetting the float-cast rule on ANY ONE array silently fails `load_npz` for the WHOLE containing fixture (Plan-Check Issue 5) — the Verify step's per-array `OracleCase::f64(name).is_some()` sweep is the guardrail; do not rely solely on the file-level `load_npz(..).is_ok()` check, which does not itself prove every individual array decoded (a `None` return from `.f64(name)` for one array can otherwise hide behind an `Ok(OracleCase)` for the rest).
- Risk: running the OvO probe AFTER already writing the fixture (rather than before) risks committing a fixture inconsistent with the actual sklearn behavior — the Implementation Steps above order the probe before generator implementation.

---

## TASK-03 — METR-CLS-01: accuracy_score (single-source with nb_common)

- **Spec:** `METR-CLS-01`
- **Order:** 2 (Wave 3a, first)
- **Depends on:** TASK-01 (types/bookkeeping), TASK-02 (fixtures)
- **Parallel with:** TASK-12 (regression track — disjoint files)

### Objective
`mlrs_algos::metrics::classification::accuracy_score(y_true, y_pred, sample_weight, normalize) -> f64`
exists and matches sklearn; `nb_common::accuracy_score` becomes a thin
re-export/delegate to it (ONE source — SPEC §5 CLS-01), and the existing NB
`score` path stays green (regression-checked, not re-implemented).

### Specification References
- `SPEC-METR-CLS-01` — weighted-fraction-of-matches contract, NB re-export requirement.

### Context and Evidence
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs:156-181` — the function being delegated to; its exact current signature is `accuracy_score(pred: &[i32], y_true: &[i32]) -> f64` (note the ARGUMENT ORDER is `pred` first, `y_true` second — opposite of sklearn's `accuracy_score(y_true, y_pred)` convention). The new `metrics::classification::accuracy_score` MUST use the sklearn-convention order `(y_true, y_pred, ...)` (SPEC §4); `nb_common::accuracy_score` keeps ITS OWN existing `(pred, y_true)` order for its one caller and internally calls the new function with arguments swapped — this order flip is the single adaptation point and must be tested explicitly (see Refactor step).
- `crates/mlrs-algos/tests/nb_common_test.rs:98-106` (`accuracy_score_fraction`) — the existing regression test that MUST still pass unchanged after the delegation (`[VERIFIED: CODEGRAPH]`).
- Fixture: `tests/fixtures/metrics_cls_binary_{f32,f64}_seed42.npz` (`ref_accuracy`, `ref_accuracy_sw`), `metrics_cls_degenerate_seed42.npz` (`ref_acc_single_match`, `ref_acc_single_mismatch`) — TASK-02.

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs` (TASK-01 already created this as a doc-comment-only stub; this task appends the first function — Plan-Check Issue 4: `metrics/mod.rs` is NOT touched by this task)
- Create: `crates/mlrs-algos/tests/metrics_classification_test.rs`
- Modify: `crates/mlrs-algos/src/naive_bayes/nb_common.rs` (delegate `accuracy_score` to the new module)

### TDD Sequence

#### 1. Red
- Test name: `accuracy_score_matches_sklearn_oracle_binary_f64` in `crates/mlrs-algos/tests/metrics_classification_test.rs`.
- Setup: load `metrics_cls_binary_f64_seed42.npz`; convert `y_true`/`y_pred` fixture floats to `i32` (fixtures store labels as float arrays per the established `expect_f64` convention — cast to `i32` on load, mirroring `fixture_vec::<F>` in `random_forest_classifier_test.rs:75-80` adapted for integer labels).
- Call `mlrs_algos::metrics::classification::accuracy_score(&y_true, &y_pred, None, true)`.
- Expected: `(got - ref_accuracy[0]).abs() < 1e-12` (EXACT tier per SPEC §6 — unweighted rational fraction).
- Expected initial failure: compile error — `mlrs_algos::metrics::classification` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test accuracy_score_matches_sklearn_oracle_binary_f64`

#### 2. Green
- Append to the existing (TASK-01-created) `classification.rs` stub: `pub fn accuracy_score(y_true: &[i32], y_pred: &[i32], sample_weight: Option<&[f64]>, normalize: bool) -> f64`: validate equal length (panic is acceptable here ONLY if mirroring the existing `nb_common` panic-on-mismatch convention exactly, OR return via a `Result` if TASK-01's bookkeeping pattern is reused — prefer reusing TASK-01's `Result<_, MetricError>`-returning bookkeeping validation for consistency, but `accuracy_score` itself may stay infallible-by-construction like the existing `nb_common` version since SPEC §4 declares its return type as bare `f64`, not `Result`). Weighted sum of matches / weighted sum (or count if `normalize=false`); `sample_weight=None` → unit weights. `weighted_correct / weighted_total` with `weighted_total = 0.0` (empty input) yields `0.0/0.0 = NaN` in IEEE-754 `f64` division WITHOUT any special-cased branch — this is the exact mechanism that preserves `nb_common`'s documented empty-input `NaN` contract (`nb_common.rs:168-171`) for free; do not add an explicit empty-input check that would change this to a different sentinel.
- `metrics/mod.rs` is NOT touched by this task (already wired by TASK-01 — Plan-Check Issue 4).
- Modify `nb_common::accuracy_score(pred: &[i32], y_true: &[i32]) -> f64` to become:
  ```rust
  pub fn accuracy_score(pred: &[i32], y_true: &[i32]) -> f64 {
      crate::metrics::classification::accuracy_score(y_true, pred, None, true)
  }
  ```
  (argument order flipped at the call site — `nb_common`'s own signature and doc-comment are UNCHANGED so its one caller, `[VERIFIED: CODEGRAPH — 1 caller]`, needs no edit).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test accuracy_score_matches_sklearn_oracle_binary_f64`

#### 3. Refactor
- Add regression assertion: `cargo test -p mlrs-algos --features cpu --test nb_common_test accuracy_score_fraction` still passes UNCHANGED (proves the delegation preserves NB's existing behavior/argument-order contract).
- Add ONE new one-line assertion test, `nb_common_accuracy_score_empty_input_is_nan` in `crates/mlrs-algos/tests/nb_common_test.rs` (Plan-Check Issue 8): `assert!(nb_common::accuracy_score(&[], &[]).is_nan())` — locks in that the new `weighted_correct/weighted_total = 0.0/0.0` division path (rather than an explicit early-return) still produces the documented `NaN` for empty input (`nb_common.rs:168-171`'s existing doc-comment contract), so a future refactor of the shared `accuracy_score` cannot silently change empty-input behavior to `0.0` or a panic without this test catching it.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test && cargo test -p mlrs-algos --features cpu --test nb_common_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Run: `cargo test -p mlrs-algos --features cpu --test nb_common_test` (regression gate — NB `score` path unchanged)
- Confirm: EXACT unweighted accuracy on the multiclass fixture too (add a companion `accuracy_score_matches_sklearn_oracle_multiclass_f64` using `metrics_cls_multiclass_f64_seed42.npz`).

### Implementation Steps
1. Write Red test #1 (binary, unweighted, f64) — fails on missing module.
2. Write Red test #2: `accuracy_score_weighted_matches_sklearn` — same fixture, `sample_weight` from the fixture, assert against `ref_accuracy_sw` at `1e-5`.
3. Write Red test #3: `accuracy_score_single_sample_degenerate` — loads `metrics_cls_degenerate_seed42.npz`, asserts `ref_acc_single_match`/`ref_acc_single_mismatch` (1.0 / 0.0 exactly).
4. Write Red test #4 (f32 variant of test #1, `atol=1e-4`), gated the SAME way as existing f32 tests (no `skip_f64_with_log` needed — f32 always runs).
5. Write Red test #5 (f64 variant with `capability::skip_f64_with_log()` early-return, mirroring `random_forest_classifier_test.rs:200-206`).
6. Implement `classification.rs::accuracy_score` + the `nb_common` delegation to pass all five at once (one Green pass).
7. Add the `nb_common_accuracy_score_empty_input_is_nan` regression assertion (Plan-Check Issue 8).
8. Run the `nb_common_test.rs` regression gate (both the pre-existing `accuracy_score_fraction` test and the new empty-input `NaN` assertion).

### Completion Criteria
- [x] All 5 Red tests fail for the stated reason (missing module) before Green.
- [x] All 5 pass after Green.
- [x] `nb_common_test.rs::accuracy_score_fraction` still passes unchanged.
- [x] `nb_common::accuracy_score`'s public signature and doc-comment are untouched (only its body changed).
- [x] `nb_common_test.rs::nb_common_accuracy_score_empty_input_is_nan` exists and passes (Plan-Check Issue 8).
- [x] `metrics/mod.rs` is untouched by this task (verified by `git diff` showing no change to that file within TASK-03's commit).

### Risks and Guardrails
- Risk: swapping `nb_common`'s internal call arguments backwards (re-introducing the original bug the delegation is meant to preserve, not introduce). Mitigation: the `nb_common_test.rs` regression run in Verify is the guardrail — it fails loudly on an argument-order mistake because its hand-built asymmetric example (`[1,1,0]` vs `[1,0,0]`) is order-sensitive only if mismatched with itself, so ALSO add one asymmetric-weight case here (Red test #2) to catch an order bug that a symmetric fixture might hide.

---

## TASK-04 — METR-CLS-02: confusion_matrix

- **Spec:** `METR-CLS-02`
- **Order:** 3 (Wave 3a)
- **Depends on:** TASK-03 (same file, sequential)

### Objective
`classification::confusion_matrix(y_true, y_pred, labels, sample_weight) -> Vec<Vec<f64>>`
matches `sklearn.metrics.confusion_matrix` exactly (unweighted) / ≤1e-5
(weighted), including the empty-class-via-explicit-`labels` and
all-one-class degenerate shapes.

### Specification References
- `SPEC-METR-CLS-02`

### Context and Evidence
- TASK-01's bookkeeping helper (unique-class discovery honoring an explicit `labels` order, including absent classes) is the direct dependency — reuse it, do not re-derive class ordering here.
- Fixtures: `metrics_cls_binary_{f32,f64}_seed42.npz` (`ref_confusion`, `ref_confusion_sw`), `metrics_cls_multiclass_{f32,f64}_seed42.npz` (`ref_confusion`), `metrics_cls_degenerate_seed42.npz` (`ref_confusion_empty` with `labels_empty`, `ref_confusion_one`).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `confusion_matrix_empty_class_via_explicit_labels` in `metrics_classification_test.rs`.
- Setup: load `metrics_cls_degenerate_seed42.npz`; `y_true_empty`/`y_pred_empty` (only classes 0,1 ever appear), `labels_empty = [0,1,2]`.
- Call `classification::confusion_matrix(&y_true_empty, &y_pred_empty, Some(&labels_empty), None)`.
- Expected: `got == ref_confusion_empty` EXACTLY (3×3, a full zero row AND column at index 2 — SPEC §6 degenerate case).
- Expected initial failure: compile error — `confusion_matrix` does not exist in `classification.rs` yet.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test confusion_matrix_empty_class_via_explicit_labels`

#### 2. Green
- Implement `confusion_matrix` using TASK-01's bookkeeping-style class resolution (sorted unique of `y_true ∪ y_pred` when `labels=None`, else `labels` verbatim including absent classes) — build the matrix by a single weighted pass counting `(true_class_idx, pred_class_idx)` pairs; rows/cols in the resolved class order.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test confusion_matrix_empty_class_via_explicit_labels`

#### 3. Refactor
- Ensure the class-resolution logic here reuses TASK-01's ALREADY-EXISTING `metrics::mod`-level bookkeeping function's class-order output directly (call it, do not re-derive class order independently) — per Plan-Check Issue 4, `metrics/mod.rs` is edited EXACTLY ONCE (in TASK-01) for the whole plan; if a genuinely new shared helper is needed beyond what TASK-01 already exposed, add it as a `pub(crate)` function INSIDE `classification.rs` instead (visible to `regression.rs` only if re-exported through the crate root, which this plan does not need), never by reopening `metrics/mod.rs`.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: all-one-class (`ref_confusion_one`, `[[n]]`), binary oracle (unweighted EXACT + weighted ≤1e-5), multiclass oracle (EXACT) all pass.

### Implementation Steps
1. Red test #1 (above).
2. Red test #2: `confusion_matrix_all_one_class` — `ref_confusion_one`.
3. Red test #3: `confusion_matrix_matches_sklearn_oracle_binary` (unweighted EXACT via `ref_confusion`, f32+f64).
4. Red test #4: `confusion_matrix_weighted_matches_sklearn_oracle_binary` (`ref_confusion_sw`, ≤1e-5).
5. Red test #5: `confusion_matrix_matches_sklearn_oracle_multiclass` (`ref_confusion`, 3×3, EXACT).
6. One Green pass implementing `confusion_matrix` to satisfy all five.

### Completion Criteria
- [x] All 5 Red tests fail (missing fn) before Green.
- [x] All 5 pass after Green.
- [x] Row/column order verified to follow `labels` exactly when given (including an absent class), sorted-unique otherwise.

### Risks and Guardrails
- Risk: counting pairs where either `y_true[i]` or `y_pred[i]` falls OUTSIDE the resolved `labels` set (sklearn silently drops such rows when `labels` is explicit and narrower than the data) — confirm this against the fixture's actual sklearn-generated reference rather than assuming; the `ref_confusion_empty` fixture (labels narrower is NOT the case here — labels is a SUPERSET) does not exercise the narrower-than-data case, so if narrowing is out of scope for the committed fixtures, do not over-implement it — note as an explicit non-goal if untested.

---

## TASK-05 — METR-CLS-03: precision_score

- **Spec:** `METR-CLS-03`
- **Order:** 4 (Wave 3a)
- **Depends on:** TASK-04

### Objective
`classification::precision_score(y_true, y_pred, labels, pos_label, average, sample_weight, zero_division) -> PrfOut`
matches sklearn for every `average` value, with `zero_division` applied
exactly on the no-predicted-positives degenerate case.

### Specification References
- `SPEC-METR-CLS-03`

### Context and Evidence
- TASK-01's per-class weighted `(tp, fp, fn)` bookkeeping is the direct input — precision per class = `tp / (tp + fp)`.
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_precision_binary`, `ref_precision_binary_sw`), `metrics_cls_multiclass_*_seed42.npz` (`ref_precision_{macro,micro,weighted,none}`, `ref_precision_macro_sw`, `ref_precision_labelreorder` + `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` — Plan-Check Issue 6), `metrics_cls_degenerate_seed42.npz` (`ref_precision_zerodiv`).
- SPEC §2/§5: "`labels` parameter for ... precision/recall/f1 ... — each with its own reorder acceptance test" — this task therefore carries its OWN `labels`-reorder Red test (Issue 6), not merely a shared fixture referenced from TASK-07.

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `precision_score_zero_division_no_predicted_positives` in `metrics_classification_test.rs`.
- Setup: load `metrics_cls_degenerate_seed42.npz`; `y_true_zp`/`y_pred_zp` (constructed so the positive class is NEVER predicted).
- Call `classification::precision_score(&y_true_zp, &y_pred_zp, None, 1, Average::Binary, None, ZeroDivision::Zero)`.
- Expected: `PrfOut::Scalar(0.0)` (matches `ref_precision_zerodiv`, EXACT — SPEC §6 zero-division degenerate).
- Expected initial failure: compile error — `precision_score` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test precision_score_zero_division_no_predicted_positives`

#### 2. Green
- Implement `precision_score` using TASK-01's bookkeeping: per-class `tp/(tp+fp)`, `zero_division` value substituted when `tp+fp==0`; `Average::Binary` selects the `pos_label` class's scalar; `Average::Macro` = unweighted mean over classes; `Average::Micro` = `sum(tp)/(sum(tp)+sum(fp))` (a SINGLE global ratio, not a per-class mean); `Average::Weighted` = mean weighted by each class's true-label support (`tp+fn`); `Average::None_` returns `PrfOut::PerClass` in the resolved class order.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test precision_score_zero_division_no_predicted_positives`

#### 3. Refactor
- Confirm the per-average dispatch is a single small `match` over `Average` reusing one shared per-class-ratio helper (so TASK-06/TASK-07 mirror it exactly, avoiding 3x duplicated average-dispatch logic — extract a private `average_ratio(per_class_ratios, supports, average, zero_division) -> PrfOut` helper in `classification.rs` if the duplication is already visible after this task).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: `average=None` per-class vector matches `ref_precision_none` elementwise; every average (`macro`,`micro`,`weighted`) matches its `ref_precision_*` fixture value.

### Implementation Steps
1. Red test #1 (above, `average='binary'` zero-division).
2. Red test #2: `precision_score_binary_matches_sklearn_oracle` (`ref_precision_binary`, EXACT-if-rational else ≤1e-5).
3. Red test #3: `precision_score_binary_weighted_matches_sklearn_oracle` (`ref_precision_binary_sw`).
4. Red test #4..#7: one per `average ∈ {macro,micro,weighted,None}` against `metrics_cls_multiclass_*_seed42.npz`'s `ref_precision_{macro,micro,weighted,none}`.
5. Red test #8: `precision_score_macro_weighted_matches_sklearn_oracle` (`ref_precision_macro_sw`).
6. Red test #9: `precision_score_labels_reorder_matches_sklearn_oracle` (Plan-Check Issue 6) — call `classification::precision_score(&y_true_labelreorder, &y_pred_labelreorder, Some(&labels_reorder), 1, Average::Macro, None, ZeroDivision::Zero)` and assert against `ref_precision_labelreorder`; this specifically exercises `labels` as a REORDER (not just a superset, which TASK-04's confusion-matrix test already covers), proving class order/contents follow `labels` verbatim through the precision computation too.
7. One Green pass for all 9.

### Completion Criteria
- [x] All 9 Red tests fail (missing fn) before Green.
- [x] All 9 pass after Green.
- [x] `average=None` returns `PrfOut::PerClass` in the resolved (sorted or `labels`) class order.
- [x] The `labels`-reorder test (Issue 6) passes, proving `labels` is honored as an explicit class ORDER, not just a superset filter.

### Risks and Guardrails
- Risk: conflating `micro` (one global ratio) with `macro` (mean of per-class ratios) — the two produce IDENTICAL results only in degenerate cases, so the multiclass fixture's distinct `ref_precision_macro` vs `ref_precision_micro` values are the guardrail that catches this if swapped.

---

## TASK-06 — METR-CLS-04: recall_score

- **Spec:** `METR-CLS-04`
- **Order:** 5 (Wave 3a)
- **Depends on:** TASK-05

### Objective
`classification::recall_score(...) -> PrfOut` mirrors TASK-05's structure with
`recall = tp / (tp + fn)` and the recall-specific zero-division degenerate
(no true positives for the class).

### Specification References
- `SPEC-METR-CLS-04`

### Context and Evidence
- Reuses the `average_ratio` helper extracted in TASK-05's Refactor step (if extracted) — recall's per-class ratio is `tp/(tp+fn)` instead of `tp/(tp+fp)`.
- Fixtures: `ref_recall_binary`, `ref_recall_binary_sw`, `ref_recall_{macro,micro,weighted,none}`, `ref_recall_macro_sw`, `ref_recall_zerodiv`, `ref_recall_labelreorder` (same `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` inputs TASK-05 used — Plan-Check Issue 6, "each with its own reorder acceptance test").

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `recall_score_zero_division_no_true_positives` — `y_true_zr`/`y_pred_zr` from `metrics_cls_degenerate_seed42.npz` (constructed so the positive class NEVER appears in `y_true` or is never correctly predicted, per SPEC §6's "no true positives" case), asserted against `ref_recall_zerodiv` (0.0).
- Expected initial failure: compile error — `recall_score` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test recall_score_zero_division_no_true_positives`

#### 2. Green
- Implement `recall_score` reusing the shared `average_ratio` dispatch with `tp/(tp+fn)` as the per-class ratio.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test recall_score_zero_division_no_true_positives`

#### 3. Refactor
- Confirm `precision_score`/`recall_score` differ ONLY in which bookkeeping field feeds the ratio denominator — no logic duplication beyond that one substitution.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: all `average` variants + weighted + zero-division match their `ref_recall_*` fixtures.

### Implementation Steps
1. Red test #1 (above).
2. Red tests #2-#8 mirroring TASK-05's structure exactly, substituting `ref_recall_*` fixture keys.
3. Red test #9: `recall_score_labels_reorder_matches_sklearn_oracle` (Plan-Check Issue 6) — same `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` inputs as TASK-05, asserted against `ref_recall_labelreorder`.
4. One Green pass.

### Completion Criteria
- [x] All 9 Red tests fail (missing fn) before Green; all pass after.
- [x] No duplicated average-dispatch logic (reuses TASK-05's helper).
- [x] The `labels`-reorder test (Issue 6) passes for `recall_score` independently of TASK-05's precision test.

### Risks and Guardrails
- Risk: same macro/micro confusion as TASK-05 — same fixture-based guardrail applies.

---

## TASK-07 — METR-CLS-05: f1_score

- **Spec:** `METR-CLS-05`
- **Order:** 6 (Wave 3a)
- **Depends on:** TASK-06

### Objective
`classification::f1_score(...) -> PrfOut` computed from the SAME weighted
`tp/fp/fn` bookkeeping directly (`f1 = 2*tp / (2*tp + fp + fn)` per class),
NOT from `precision_score(...) × recall_score(...)` floats (SPEC §5 CLS-05
note — avoids double-rounding).

### Specification References
- `SPEC-METR-CLS-05`

### Context and Evidence
- SPEC §5: "f1 is computed from the same weighted TP/FP/FN (harmonic mean), NOT from mlrs precision×recall floats, to avoid double-rounding" — this is the task's PRINCIPAL implementation constraint, verified by a dedicated Red test (see below) that would catch a precision×recall-derived implementation under floating-point rounding.
- Fixtures: `ref_f1_binary`, `ref_f1_binary_sw`, `ref_f1_{macro,micro,weighted,none}`, `ref_f1_macro_sw`, `ref_f1_zerodiv`, `ref_f1_labelreorder` (same `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` inputs as TASK-05/06 — Plan-Check Issue 6).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `f1_score_computed_from_tp_fp_fn_not_precision_times_recall` — hand-built case with a `tp/fp/fn` combination known to produce a DIFFERENT f1 value under `2*tp/(2*tp+fp+fn)` vs. a naive `2*P*R/(P+R)` computed from independently-rounded `P`/`R` floats (a case with a non-terminating binary fraction in `P`/`R` individually but a cleaner value in the direct formula, e.g. `tp=1,fp=2,fn=0` at `f32`).
- Expected: the direct-formula value to `1e-7` (f64) tightness — tighter than the general ≤1e-5 gate, specifically to catch the double-rounding regression.
- Expected initial failure: compile error — `f1_score` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test f1_score_computed_from_tp_fp_fn_not_precision_times_recall`

#### 2. Green
- Implement `f1_score` with its OWN per-class ratio `2*tp/(2*tp+fp+fn)` fed through the shared `average_ratio` dispatch — do NOT call `precision_score`/`recall_score` internally.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test f1_score_computed_from_tp_fp_fn_not_precision_times_recall`

#### 3. Refactor
- Code-review check (not a test): grep the `f1_score` implementation body to confirm it does not call `precision_score(...)`/`recall_score(...)`.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: all oracle + zero-division + weighted variants match `ref_f1_*`.

### Implementation Steps
1. Red test #1 (above, the double-rounding guard).
2. Red test #2: `f1_score_zero_division_degenerate` (`ref_f1_zerodiv`).
3. Red tests #3-#9 mirroring TASK-05/06's oracle + average + weighted structure against `ref_f1_*`.
4. Red test #10: `f1_score_labels_reorder_matches_sklearn_oracle` (Plan-Check Issue 6) — same `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` inputs as TASK-05/06, asserted against `ref_f1_labelreorder`.
5. One Green pass.

### Completion Criteria
- [x] The double-rounding guard test fails (missing fn) before Green, passes after.
- [x] All oracle/zero-division/average/weighted tests pass.
- [x] `f1_score`'s implementation does not call `precision_score`/`recall_score`.
- [x] The `labels`-reorder test (Issue 6) passes for `f1_score`, completing the "each with its own reorder acceptance test" requirement across TASK-05/06/07.

### Risks and Guardrails
- Risk: reviewer intuition says "compute f1 from P and R" — the double-rounding Red test is the explicit guardrail against this natural-but-wrong shortcut.

---

## TASK-08 — METR-CLS-06: log_loss

- **Spec:** `METR-CLS-06`
- **Order:** 7 (Wave 3a)
- **Depends on:** TASK-07

### Objective
`classification::log_loss(y_true, y_prob, n_classes, labels, sample_weight, eps, normalize) -> f64`
matches sklearn's weighted cross-entropy with `[eps, 1-eps]` clipping,
including the `0.0`/`1.0`-probability clipping degenerate.

### Specification References
- `SPEC-METR-CLS-06`

### Context and Evidence
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs:72` `log_sum_exp_normalize` is a NEARBY but NOT directly reusable helper (it operates on log-likelihoods pre-normalization, not on already-normalized clipped probabilities) — `log_loss` clips + takes `ln` directly on the (renormalized) probability the caller supplies, it does not need the log-sum-exp trick. Do not force-reuse `log_sum_exp_normalize` if the shapes do not match; implement the clip+`ln` directly (RESEARCH-METRICS §3 flags this as "directly relevant" contextually, not as a literal call site).
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_log_loss_binary` + `y_prob_binary` — Plan-Check Issue 7, closes SPEC §5 METR-CLS-06's "binary + multiclass" wording), `metrics_cls_multiclass_*_seed42.npz` (`ref_log_loss`, `ref_log_loss_sw`), `metrics_cls_degenerate_seed42.npz` (`y_true_clip`/`y_prob_clip`/`ref_log_loss_clip`, `y_true_logloss_labelreorder`/`y_prob_logloss_labelreorder`/`labels_logloss_reorder`/`ref_log_loss_labelreorder` — Plan-Check Issue 6).
- `[UNVERIFIED]`: whether sklearn's installed `log_loss` (1.9.0 per Q6) renormalizes each row to sum to 1 AFTER clipping, or only clips without renormalizing. TASK-02's Green step must read the ACTUAL fixture value produced by the pinned sklearn version; this task's Green implementation must be checked against that stored value, not a memorized formula. Flag this explicitly in code review before merging.
- SPEC §2: "`labels` parameter for ... log_loss (reorder + subset) — each with its own reorder acceptance test" — this task carries its OWN `labels`-reorder Red test (Issue 6).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `log_loss_clips_zero_and_one_probabilities_to_finite_value` — `y_true_clip`/`y_prob_clip` (containing exact `0.0`/`1.0` entries) from `metrics_cls_degenerate_seed42.npz`.
- Call `classification::log_loss(&y_true_clip, &y_prob_clip, n_classes, None, None, 1e-15, true)`.
- Expected: a FINITE value equal to `ref_log_loss_clip` within `1e-5` (SPEC §6 clipping degenerate).
- Expected initial failure: compile error — `log_loss` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test log_loss_clips_zero_and_one_probabilities_to_finite_value`

#### 2. Green
- Implement `log_loss`: clip every `y_prob` entry to `[eps, 1-eps]`; compute per-row `-ln(p[true_class])`; weighted mean (or sum if `normalize=false`) across rows. Attempt WITHOUT renormalization first; if the fixture assertion (`ref_log_loss`/`ref_log_loss_clip`) does not match within tolerance, add per-row renormalization after clipping and re-check — resolve empirically against the committed oracle value (this is the concrete resolution of the `[UNVERIFIED]` flag above).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test log_loss_clips_zero_and_one_probabilities_to_finite_value`

#### 3. Refactor
- Document (in a doc-comment on `log_loss`) which of the two behaviors (renormalize-after-clip or clip-only) matched, citing the fixture that pinned it — closes the `[UNVERIFIED]` flag with evidence for future readers.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: multiclass oracle (`ref_log_loss`, ≤1e-5) + weighted (`ref_log_loss_sw`) + clipping degenerate all pass.

### Implementation Steps
1. Red test #1 (above, clipping degenerate).
2. Red test #2: `log_loss_matches_sklearn_oracle_multiclass` (`ref_log_loss`, ≤1e-5, f32+f64).
3. Red test #3: `log_loss_weighted_matches_sklearn_oracle` (`ref_log_loss_sw`).
4. Red test #4: `log_loss_matches_sklearn_oracle_binary` (Plan-Check Issue 7) — `ref_log_loss_binary` against `y_true`/`y_prob_binary` from `metrics_cls_binary_*_seed42.npz`, closing SPEC §5 METR-CLS-06's explicit "binary + multiclass" acceptance wording (the multiclass fixture mathematically subsumes binary, but SPEC calls for an explicit binary case).
5. Red test #5: `log_loss_labels_reorder_matches_sklearn_oracle` (Plan-Check Issue 6) — `classification::log_loss(&y_true_logloss_labelreorder, &y_prob_logloss_labelreorder, 2, Some(&labels_logloss_reorder), None, 1e-15, true)` against `ref_log_loss_labelreorder`, proving the `labels` parameter reorders which probability COLUMN is treated as which class.
6. One Green pass resolving the renormalization question empirically.

### Completion Criteria
- [x] All 5 Red tests fail (missing fn) before Green.
- [x] All 5 pass after Green.
- [x] The renormalize-vs-clip-only behavior is documented with its evidence source (the fixture that discriminated it).
- [x] An explicit binary `log_loss` oracle test exists (Issue 7) and the `labels`-reorder test exists (Issue 6).

### Risks and Guardrails
- Risk (highest in this task): the `[UNVERIFIED]` clipping/renormalization detail could silently pass the multiclass fixture (large row count, small resulting error) while failing the TIGHT clipping-degenerate fixture (rows constructed exactly to expose it) — keep the clipping-degenerate test's tolerance at the SAME ≤1e-5 gate, do not loosen it to make it pass.

---

## TASK-09 — METR-CLS-07: roc_auc_score (binary)

- **Spec:** `METR-CLS-07`
- **Order:** 8 (Wave 3a)
- **Depends on:** TASK-08

### Objective
`classification::roc_auc_score_binary(y_true, y_score, pos_label, sample_weight) -> Result<f64, MetricError>`
matches sklearn's rank-based AUC (tie-heavy ranks handled), returning
`Err(MetricError::SingleClassRocAuc)` when only one class is present.

### Specification References
- `SPEC-METR-CLS-07`

### Context and Evidence
- Implementation strategy (Planner's choice, not a copied sklearn internal): threshold-sweep — sort samples by DESCENDING score (stable sort, so ties keep original order for reproducibility), accumulate weighted TP/FP counts as the threshold sweeps past each distinct score value (average-rank tie handling: samples sharing an exact score value are grouped into ONE step so ties do not artificially separate), normalize by total weighted positives/negatives to get TPR/FPR, then trapezoidal-integrate (`Σ (fpr[i]-fpr[i-1]) * (tpr[i]+tpr[i-1])/2`). This generalizes correctly to both the weighted and unweighted case (SPEC §4 `roc_auc_score_binary` returns `Result` — first `Result`-returning metric fn in this module, matching `MetricError::SingleClassRocAuc`).
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_roc_auc`, `ref_roc_auc_sw`, tie-heavy `y_score`), `metrics_cls_degenerate_seed42.npz` (`y_true_singleclass`/`y_score_singleclass` — no ref value, an error gate).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `roc_auc_score_binary_single_class_returns_error` — `y_true_singleclass = [1,1,1]` (all one class), any `y_score_singleclass`.
- Call `classification::roc_auc_score_binary(&y_true_singleclass, &y_score_singleclass, 1, None)`.
- Expected: `Err(MetricError::SingleClassRocAuc)` — the error GATE, not a value (SPEC §6: "gate the error, not a value").
- Expected initial failure: compile error — `roc_auc_score_binary` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test roc_auc_score_binary_single_class_returns_error`

#### 2. Green
- Implement `roc_auc_score_binary` per the threshold-sweep strategy above; the single-class precondition check happens BEFORE the sweep (fewer than 2 distinct classes in `y_true` → `Err(MetricError::SingleClassRocAuc)`).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test roc_auc_score_binary_single_class_returns_error`

#### 3. Refactor
- Extract the "sort-by-descending-score with weighted cumulative TP/FP" sweep as a private helper reusable by TASK-10 (multiclass OvR/OvO reduces to repeated binary sweeps) and TASK-11 (`precision_recall_curve` needs the same sorted-cumulative-count machinery).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: tie-heavy binary oracle (`ref_roc_auc`, ≤1e-5) + weighted (`ref_roc_auc_sw`) both pass.

### Implementation Steps
1. Red test #1 (above, single-class error gate).
2. Red test #2: `roc_auc_score_binary_matches_sklearn_oracle_tie_heavy` (`ref_roc_auc`, ≤1e-5).
3. Red test #3: `roc_auc_score_binary_weighted_matches_sklearn_oracle` (`ref_roc_auc_sw`).
4. One Green pass; extract the shared sweep helper in Refactor.

### Completion Criteria
- [x] All 3 Red tests fail (missing fn) before Green.
- [x] All 3 pass after Green.
- [x] A reusable sweep helper exists for TASK-10/TASK-11 to consume (not a hard requirement to complete THIS task's specs, but MUST exist before TASK-10/11 start, per this task's Refactor step).

### Risks and Guardrails
- Risk: naive rank-sum (Mann-Whitney U) formula is simpler but does NOT generalize cleanly to `sample_weight` — the threshold-sweep + trapezoidal strategy is chosen specifically because it is correct for both; do not fall back to the simpler unweighted-only formula even for the unweighted test, to avoid a second, divergent code path.

---

## TASK-10 — METR-CLS-08: roc_auc_score (multiclass OvR/OvO)

- **Spec:** `METR-CLS-08`
- **Order:** 9 (Wave 3a)
- **Depends on:** TASK-09

### Objective
`classification::roc_auc_score_multiclass(y_true, y_score, n_classes, multi_class, average, sample_weight) -> Result<f64, MetricError>`
matches sklearn for every `(multi_class, average)` combination in
`{ovr,ovo} × {macro,weighted}`, on BOTH unweighted and OvR-weighted inputs
(Plan-Check Issue 1), with the OvO+`sample_weight` carve-out (Plan-Check
Issue 2 / SPEC §2 Q10) implemented as one of two mutually exclusive branches
depending on what TASK-02's probe found.

### Specification References
- `SPEC-METR-CLS-08`

### Context and Evidence
- Reuses TASK-09's extracted sweep helper: OvR reduces to `n_classes` independent binary AUCs (class `c`'s score column vs. `y_true==c`), macro-averaged (unweighted mean) or weighted (weighted by each class's true-label prevalence); `sample_weight` flows straight through to each per-class binary-AUC call (TASK-09's helper already supports it — no OvR carve-out). OvO averages the pairwise AUC over all `C(n_classes,2)` class pairs — each pair's AUC computed bidirectionally (class `i` as positive vs. `j`, and `j` as positive vs. `i`) and averaged, per sklearn's Hand & Till formulation; `weighted` OvO weights each pair by its prevalence.
- **OvO + `sample_weight` carve-out (SPEC §2/§4 revision, Plan-Check Issue 2):** TASK-02's Green step PROBED whether `scikit-learn==1.9.0`'s `roc_auc_score(multi_class='ovo', sample_weight=...)` raises. This task's Green step branches on that recorded outcome:
  - **Branch A (probe found sklearn RAISES):** the OvO arm of `roc_auc_score_multiclass` returns `Err(MetricError::WeightedOvoUnsupported)` immediately whenever `multi_class == MultiClass::Ovo && sample_weight.is_some()`, BEFORE running any pairwise sweep — matching sklearn's own rejection. No weighted-OvO VALUE fixture exists in this branch (TASK-02 did not generate one).
  - **Branch B (probe found sklearn does NOT raise):** the OvO arm accepts `sample_weight` and applies it exactly like the OvR arm (weight flows into each pairwise binary-AUC call); a weighted-OvO VALUE fixture (`ref_roc_auc_ovo_macro_sw`/`ref_roc_auc_ovo_weighted_sw`) exists and is asserted against.
  - Exactly ONE of these two branches is implemented, per what TASK-02 actually recorded — do not implement both defensively; read TASK-02's generator docstring (which states the probe outcome) before writing this task's Green code.
- Fixtures: `metrics_cls_multiclass_*_seed42.npz` (`ref_roc_auc_ovr_macro`, `ref_roc_auc_ovr_weighted`, `ref_roc_auc_ovo_macro`, `ref_roc_auc_ovo_weighted`, `ref_roc_auc_ovr_macro_sw`, `ref_roc_auc_ovr_weighted_sw` — Issue 1, always present — and, ONLY in Branch B, `ref_roc_auc_ovo_macro_sw`/`ref_roc_auc_ovo_weighted_sw`).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `roc_auc_score_multiclass_ovr_macro_matches_sklearn_oracle`.
- Call `classification::roc_auc_score_multiclass(&y_true, &y_proba_flat, 3, MultiClass::Ovr, Average::Macro, None)`.
- Expected: `Ok(v)` where `(v - ref_roc_auc_ovr_macro).abs() <= 1e-5`.
- Expected initial failure: compile error — `roc_auc_score_multiclass` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test roc_auc_score_multiclass_ovr_macro_matches_sklearn_oracle`

#### 2. Green
- Implement `roc_auc_score_multiclass` dispatching on `multi_class`: `Ovr` calls TASK-09's binary AUC helper per class column (with `sample_weight` passed straight through — no carve-out); `Ovo` iterates class pairs calling the same helper bidirectionally per pair, with the Branch-A/Branch-B carve-out described above gating whether `sample_weight.is_some()` short-circuits to `Err(MetricError::WeightedOvoUnsupported)` or flows into the pairwise sweep. `average` selects unweighted-mean vs. prevalence-weighted-mean over the per-class (OvR) or per-pair (OvO) AUC values.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test roc_auc_score_multiclass_ovr_macro_matches_sklearn_oracle`

#### 3. Refactor
- Confirm no duplicated sweep logic — both OvR and OvO call TASK-09's single binary-AUC helper; only the class-column/pair selection, the OvO weight carve-out branch, and the final averaging differ.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Confirm: all 4 unweighted `(multi_class, average)` combinations match their `ref_roc_auc_{ovr,ovo}_{macro,weighted}` fixtures; the 2 weighted-OvR combinations match `ref_roc_auc_ovr_{macro,weighted}_sw`; the OvO+weight case matches EITHER the `Err(MetricError::WeightedOvoUnsupported)` gate (Branch A) OR `ref_roc_auc_ovo_{macro,weighted}_sw` (Branch B), whichever TASK-02 determined.

### Implementation Steps
1. Red test #1 (above, OvR macro, unweighted).
2. Red test #2: `roc_auc_score_multiclass_ovr_weighted_matches_sklearn_oracle`.
3. Red test #3: `roc_auc_score_multiclass_ovo_macro_matches_sklearn_oracle` (unweighted).
4. Red test #4: `roc_auc_score_multiclass_ovo_weighted_matches_sklearn_oracle` (unweighted, `average='weighted'`).
5. Red test #5: `roc_auc_score_multiclass_ovr_weighted_sample_weight_matches_sklearn_oracle` (Plan-Check Issue 1) — `ref_roc_auc_ovr_macro_sw`/`ref_roc_auc_ovr_weighted_sw`, both `average` values with `sample_weight` set.
6. Red test #6: `roc_auc_score_multiclass_ovo_weighted_sample_weight_gate` (Plan-Check Issue 2) — EITHER `assert!(matches!(classification::roc_auc_score_multiclass(&y_true, &y_proba_flat, 3, MultiClass::Ovo, Average::Macro, Some(&sample_weight)), Err(MetricError::WeightedOvoUnsupported)))` (Branch A) OR a value assertion against `ref_roc_auc_ovo_macro_sw` (Branch B) — write whichever branch TASK-02's generator docstring recorded; do NOT write both.
7. One Green pass for all 6.

### Completion Criteria
- [x] All 6 Red tests fail (missing fn) before Green.
- [x] All 6 pass after Green.
- [x] Implementation calls TASK-09's binary-AUC helper — no re-implemented sweep.
- [x] The OvO+`sample_weight` carve-out branch actually implemented matches the ONE branch TASK-02's probe recorded (not both, not neither).
- [x] OvR's `sample_weight` support has NO carve-out (weighted OvR always computes a value, never an error).

### Risks and Guardrails
- Risk: OvO's bidirectional pairwise averaging is the single most error-prone formula in this plan (SPEC §9 risk 4) — the fixture cross-product (2 `multi_class` × 2 `average`, unweighted + OvR-weighted) is the guardrail; do not consider this task done until all combinations pass independently (a bug in only the OvO branch could still pass OvR's fixtures).
- Risk: implementing BOTH the error-gate AND a weighted-OvO value path "just in case" would silently diverge from what TASK-02 actually generated (only one of the two fixtures/gates exists) — read TASK-02's recorded probe outcome FIRST, implement only the matching branch.

---

## TASK-11 — METR-CLS-09: precision_recall_curve

- **Spec:** `METR-CLS-09`
- **Order:** 10 (Wave 3a, last)
- **Depends on:** TASK-10

### Objective
`classification::precision_recall_curve(y_true, probas_pred, pos_label, sample_weight) -> (Vec<f64>, Vec<f64>, Vec<f64>)`
matches sklearn's threshold sweep: `precision`/`recall` length =
`thresholds.len()+1`, trailing `(1.0, 0.0)` sentinel, ascending thresholds.

### Specification References
- `SPEC-METR-CLS-09`

### Context and Evidence
- Reuses TASK-09's sorted-cumulative-weighted-count sweep machinery (same underlying sort-by-score + cumulative TP/FP the ROC sweep uses), redirected to precision/recall instead of TPR/FPR.
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_pr_precision`, `ref_pr_recall`, `ref_pr_thresholds`, `ref_pr_precision_sw`, `ref_pr_recall_sw`, `ref_pr_thresholds_sw` — Plan-Check Issue 1, ALWAYS generated by TASK-02, so the weighted test below is a REQUIRED Red test, not a conditional one).

### Files
- Modify: `crates/mlrs-algos/src/metrics/classification.rs`
- Modify: `crates/mlrs-algos/tests/metrics_classification_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `precision_recall_curve_sentinel_and_length_invariants`.
- Call `classification::precision_recall_curve(&y_true, &probas_pred, 1, None)` on the tie-heavy binary fixture.
- Expected: `precision.len() == thresholds.len() + 1`, `recall.len() == thresholds.len() + 1`, `precision.last() == Some(&1.0)`, `recall.last() == Some(&0.0)`, `thresholds` strictly non-decreasing (ascending).
- Expected initial failure: compile error — `precision_recall_curve` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test precision_recall_curve_sentinel_and_length_invariants`

#### 2. Green
- Implement `precision_recall_curve`: sort samples by DESCENDING score, sweep distinct score values (ascending emission order for `thresholds`, so internally iterate the descending-sorted sweep and reverse before returning, OR iterate ascending directly — whichever keeps the cumulative-count math simplest), computing `precision = tp/(tp+fp)`, `recall = tp/P` (P = total weighted positives) at each distinct threshold, then append the `(1.0, 0.0)` sentinel.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test precision_recall_curve_sentinel_and_length_invariants`

#### 3. Refactor
- Confirm this function reuses TASK-09's sort/sweep helper rather than re-sorting independently.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test`
- Run the FULL classification suite as the Wave-3a regression gate: `cargo test -p mlrs-algos --features cpu --test metrics_classification_test` (all tasks TASK-03..11's tests together) and `cargo test -p mlrs-algos --features wgpu --test metrics_classification_test` (f32 gate, per SOURCES.md validation commands).
- Confirm: elementwise `precision`/`recall`/`thresholds` match `ref_pr_precision`/`ref_pr_recall`/`ref_pr_thresholds` at ≤1e-5, both trivial and tie-heavy fixtures.

### Implementation Steps
1. Red test #1 (above, structural invariants).
2. Red test #2: `precision_recall_curve_matches_sklearn_oracle_tie_heavy` (elementwise ≤1e-5 against `ref_pr_precision`/`ref_pr_recall`/`ref_pr_thresholds`).
3. Red test #3: `precision_recall_curve_weighted_matches_sklearn_oracle` (Plan-Check Issue 1 — REQUIRED, not conditional: call `classification::precision_recall_curve(&y_true, &probas_pred, 1, Some(&sample_weight))` and assert elementwise ≤1e-5 against `ref_pr_precision_sw`/`ref_pr_recall_sw`/`ref_pr_thresholds_sw`, which TASK-02 always generates).
4. One Green pass.
5. Run the full Wave 3a regression suite (all of TASK-03..11's tests) on BOTH `cpu` and `wgpu` features.

### Completion Criteria
- [x] All Red tests fail (missing fn) before Green.
- [x] All pass after Green.
- [x] The weighted `precision_recall_curve` test (Issue 1) passes — `sample_weight` is exercised on this metric, closing the previously-untested locked requirement.
- [x] The full `metrics_classification_test.rs` suite (TASK-03..11 combined) passes on `--features cpu` AND `--features wgpu`.

### Risks and Guardrails
- Risk: sentinel/length invariants are easy to get right in isolation but wrong when combined with tie-grouping (a tie-heavy fixture with `k` distinct score values must produce EXACTLY `k+1` precision/recall entries, not `n+1` where `n` is the raw sample count) — the tie-heavy fixture is the guardrail.

---

## TASK-12 — METR-REG-01: r2_score

- **Spec:** `METR-REG-01`
- **Order:** 2 (Wave 3b, first — parallel with TASK-03)
- **Depends on:** TASK-01 (types), TASK-02 (fixtures)
- **Parallel with:** TASK-03..11 (disjoint files)

### Objective
`regression::r2_score<F: Float>(y_true, y_pred, sample_weight) -> f64` matches
sklearn, including the constant-target denominator-zero degenerate pinned to
the ACTUAL sklearn-produced fixture value (not hand-derived), and perfect
prediction (`r2=1.0`).

### Specification References
- `SPEC-METR-REG-01`

### Context and Evidence
- `crates/mlrs-algos/src/covariance/empirical_covariance.rs:414-427` — the f64-accumulate-then-cast precedent this module follows for numeric stability regardless of input `F` (`[VERIFIED: LOCAL]`, cited in SPEC §3/§4: "generic over input float, accumulate f64").
- Fixture: `metrics_reg_{f32,f64}_seed42.npz` (`ref_r2`, `ref_r2_sw`, `y_true_const`/`y_pred_const`/`ref_r2_const`, `y_perfect`/`ref_r2_perfect`) — 1-D single-output only (SPEC §2 revision: multioutput is a non-goal; no 2-D array anywhere in this fixture).

### Files
- Modify: `crates/mlrs-algos/src/metrics/regression.rs` (TASK-01 already created this as a doc-comment-only stub; this task appends the first function — Plan-Check Issue 4: `metrics/mod.rs` is NOT touched by this task)
- Create: `crates/mlrs-algos/tests/metrics_regression_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `r2_score_constant_target_pins_sklearn_actual_value` in `metrics_regression_test.rs`.
- Setup: load `metrics_reg_f64_seed42.npz`; `y_true_const` (all-equal), `y_pred_const`.
- Call `regression::r2_score::<f64>(&y_true_const, &y_pred_const, None)`.
- Expected: `(got - ref_r2_const[0]).abs() <= 1e-5` — asserting against the FIXTURE value, not a hand-derived constant (SPEC §5 REG note, SPEC §9 risk 5).
- Expected initial failure: compile error — `regression::r2_score` does not exist yet (the `mlrs_algos::metrics::regression` MODULE already exists as an empty stub from TASK-01, so this is a missing-function error, not a missing-module one).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test r2_score_constant_target_pins_sklearn_actual_value`

#### 2. Green
- Append to the existing (TASK-01-created) `regression.rs` stub: `pub fn r2_score<F: Float + CubeElement + Pod>(y_true: &[F], y_pred: &[F], sample_weight: Option<&[f64]>) -> f64` (1-D `&[F]` only — no `multioutput` parameter, SPEC §2 non-goal): accumulate `ss_res = Σ w_i*(y_true_i - y_pred_i)^2` and `ss_tot = Σ w_i*(y_true_i - weighted_mean(y_true))^2` in `f64` regardless of `F`; `1.0 - ss_res/ss_tot`, with the `ss_tot==0` branch returning whatever value makes the constant-target test pass (read off the fixture — do not hand-derive per the risk note).
- `metrics/mod.rs` is NOT touched by this task (already wired by TASK-01 — Plan-Check Issue 4).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test r2_score_constant_target_pins_sklearn_actual_value`

#### 3. Refactor
- Factor the `f64`-accumulate weighted-mean/weighted-sum-of-squares helper for reuse by TASK-13/TASK-14 (MSE/MAE need the same weighted-accumulation pattern, different reduction).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`
- Confirm: standard oracle (f32+f64, ≤1e-5/atol1e-4), weighted, perfect-prediction (`r2=1.0` EXACT or ≤1e-5) all pass.

### Implementation Steps
1. Red test #1 (above, constant-target).
2. Red test #2: `r2_score_perfect_prediction_is_one` (`ref_r2_perfect`).
3. Red test #3: `r2_score_matches_sklearn_oracle_f64` (`ref_r2`, ≤1e-5).
4. Red test #4: `r2_score_matches_sklearn_oracle_f32` (`ref_r2`, `atol=1e-4`).
5. Red test #5: `r2_score_weighted_matches_sklearn_oracle` (`ref_r2_sw`).
6. One Green pass for all 5.

### Completion Criteria
- [x] All 5 Red tests fail (missing function) before Green.
- [x] All 5 pass after Green.
- [x] Constant-target value is read from the fixture, not hand-derived (verify by reading the Green implementation's comment citing the fixture).
- [x] `r2_score`'s signature takes 1-D `&[F]` only — no `multioutput` parameter (SPEC §2 non-goal).
- [x] `metrics/mod.rs` is untouched by this task (verified by `git diff` showing no change to that file within TASK-12's commit).

### Risks and Guardrails
- Risk: SPEC §9 risk 5 — hand-deriving the constant-target r2 value from memory of sklearn's source rather than the pinned fixture is the exact failure mode this task's Red test #1 exists to prevent.

---

## TASK-13 — METR-REG-02: mean_squared_error

- **Spec:** `METR-REG-02`
- **Order:** 3 (Wave 3b)
- **Depends on:** TASK-12

### Objective
`regression::mean_squared_error<F: Float>(y_true, y_pred, sample_weight) -> f64`
returns MSE ONLY — no `squared=` parameter (SPEC §2 non-goal, SPEC §9 risk 1;
sklearn ≥1.4 removed `squared=False`, RMSE is the separate
`root_mean_squared_error`, out of scope here).

### Specification References
- `SPEC-METR-REG-02`

### Context and Evidence
- SPEC §2 non-goals: "`root_mean_squared_error` / `mean_squared_error(squared=False)`... MSE-only here" (`[VERIFIED: WEB scikit-learn mean_squared_error docs]`).
- Fixture: `metrics_reg_{f32,f64}_seed42.npz` (`ref_mse`, `ref_mse_sw`, `y_perfect`/`ref_mse_perfect`).

### Files
- Modify: `crates/mlrs-algos/src/metrics/regression.rs`
- Modify: `crates/mlrs-algos/tests/metrics_regression_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `mean_squared_error_perfect_prediction_is_zero`.
- Call `regression::mean_squared_error::<f64>(&y_perfect, &y_perfect, None)`.
- Expected: `0.0` EXACTLY (`ref_mse_perfect`).
- Expected initial failure: compile error — `mean_squared_error` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test mean_squared_error_perfect_prediction_is_zero`

#### 2. Green
- Implement `mean_squared_error` reusing TASK-12's weighted-accumulate helper: `Σ w_i*(y_true_i - y_pred_i)^2 / Σ w_i`, `f64` accumulation regardless of `F`. Signature takes NO `squared` parameter (SPEC constraint — do not add one).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test mean_squared_error_perfect_prediction_is_zero`

#### 3. Refactor
- Confirm no `squared`/`root_mean_squared_error` symbol exists anywhere in `regression.rs` (grep check).
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`
- Confirm: standard oracle (f32+f64) + weighted match `ref_mse`/`ref_mse_sw`.

### Implementation Steps
1. Red test #1 (above).
2. Red test #2: `mean_squared_error_matches_sklearn_oracle_f64` (`ref_mse`, ≤1e-5).
3. Red test #3: `mean_squared_error_matches_sklearn_oracle_f32` (`atol=1e-4`).
4. Red test #4: `mean_squared_error_weighted_matches_sklearn_oracle` (`ref_mse_sw`).
5. One Green pass.

### Completion Criteria
- [x] All 4 Red tests fail (missing fn) before Green; all pass after.
- [x] No `squared` parameter anywhere in the signature.

### Risks and Guardrails
- Risk: reviewer/implementer habit of adding `squared: bool` from memory of older sklearn — SPEC explicitly forbids it; grep-check in Refactor is the guardrail.

---

## TASK-14 — METR-REG-03: mean_absolute_error

- **Spec:** `METR-REG-03`
- **Order:** 4 (Wave 3b, last)
- **Depends on:** TASK-13

### Objective
`regression::mean_absolute_error<F: Float>(y_true, y_pred, sample_weight) -> f64`
matches sklearn, including perfect prediction (`mae=0.0`).

### Specification References
- `SPEC-METR-REG-03`

### Context and Evidence
- Reuses TASK-12's weighted-accumulate helper (same pattern, `abs()` instead of squared difference).
- Fixture: `metrics_reg_{f32,f64}_seed42.npz` (`ref_mae`, `ref_mae_sw`, `ref_mae_perfect`).

### Files
- Modify: `crates/mlrs-algos/src/metrics/regression.rs`
- Modify: `crates/mlrs-algos/tests/metrics_regression_test.rs`

### TDD Sequence

#### 1. Red
- Test name: `mean_absolute_error_perfect_prediction_is_zero`.
- Call `regression::mean_absolute_error::<f64>(&y_perfect, &y_perfect, None)`.
- Expected: `0.0` EXACTLY (`ref_mae_perfect`).
- Expected initial failure: compile error — `mean_absolute_error` does not exist.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test mean_absolute_error_perfect_prediction_is_zero`

#### 2. Green
- Implement `mean_absolute_error`: `Σ w_i*|y_true_i - y_pred_i| / Σ w_i`, `f64` accumulation.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test mean_absolute_error_perfect_prediction_is_zero`

#### 3. Refactor
- Confirm `r2_score`/`mean_squared_error`/`mean_absolute_error` share ONE weighted-accumulate helper (three reduction kernels: squared-error-sum-and-variance, squared-error-sum, abs-error-sum), not three independently-written accumulation loops.
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`

#### 4. Verify
- Run: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test`
- Run the FULL Wave-3b regression gate on both `cpu` and `wgpu` features: `cargo test -p mlrs-algos --features cpu --test metrics_regression_test && cargo test -p mlrs-algos --features wgpu --test metrics_regression_test`.

### Implementation Steps
1. Red test #1 (above).
2. Red test #2: `mean_absolute_error_matches_sklearn_oracle_f64` (`ref_mae`, ≤1e-5).
3. Red test #3: `mean_absolute_error_matches_sklearn_oracle_f32` (`atol=1e-4`).
4. Red test #4: `mean_absolute_error_weighted_matches_sklearn_oracle` (`ref_mae_sw`).
5. One Green pass.
6. Run the full Wave-3b suite on both backend features.

### Completion Criteria
- [x] All 4 Red tests fail (missing fn) before Green; all pass after.
- [x] The full `metrics_regression_test.rs` suite (TASK-12..14) passes on `--features cpu` AND `--features wgpu`.

### Risks and Guardrails
- Risk: none metric-specific beyond the shared f64-accumulation discipline already tested in TASK-12/13.

---

## TASK-15 — METR-BIND-01: PyO3 free-function surface

- **Spec:** `METR-BIND-01`
- **Order:** 5 (Wave 4)
- **Depends on:** TASK-11 (all classification algos), TASK-14 (all regression algos)

### Objective
`crates/mlrs-py/src/metrics.rs` exposes one `#[pyfunction]` per algos function
(plus the `_per_class` split for `average=None` — see the resolved-decisions
section), registered in `_mlrs` via `m.add_function(wrap_pyfunction!(...))`;
`MetricError` maps to `PyValueError`; a length-mismatch input raises
`ValueError`.

### Specification References
- `SPEC-METR-BIND-01`

### Context and Evidence
- `crates/mlrs-py/src/estimators/projection.rs:379-382` (`johnson_lindenstrauss_min_dim`) — the exact `#[pyfunction]` + `algo_err_to_py`-mapping pattern to mirror (`[VERIFIED: CODEGRAPH]`).
- `crates/mlrs-py/src/lib.rs:166-169,196,238` — `backend_supports_f64` `#[pyfunction]` + its two registration call sites (`m.add_function(wrap_pyfunction!(backend_supports_f64, m)?)?;` at line 196; `m.add_function(wrap_pyfunction!(johnson_lindenstrauss_min_dim, m)?)?;` at line 238) — the registration pattern this task's `lib.rs` edit follows (`[VERIFIED: CODEGRAPH]`).
- `crates/mlrs-py/src/errors.rs:56-58` (`algo_err_to_py`) — maps `AlgoError -> PyValueError`; this task needs an ANALOGOUS `metric_err_to_py(MetricError) -> PyErr` function since `MetricError` (TASK-01) is a distinct type from `AlgoError` — add it to `crates/mlrs-py/src/errors.rs` alongside `algo_err_to_py`, following the SAME `PyValueError::new_err(err.to_string())` pattern (`[VERIFIED: CODEGRAPH]`).
- `crates/mlrs-py/Cargo.toml:44-50` — `pyo3` dev-dependency `auto-initialize` links a real interpreter for `cargo test -p mlrs-py`, so this is a genuine Rust integration test target, not just a compile check (`[VERIFIED: LOCAL]`).
- `crates/mlrs-py/tests/test_naive_bayes.py:1-58` — the sibling FFI-smoke-test convention this task's `test_metrics.py` follows STRUCTURALLY (import-guarded via `pytest.importorskip("mlrs._mlrs")`, `[VERIFIED: CODEGRAPH]`), but WITHOUT the `pyarrow`/`_arrow()` helper — metrics take plain Python lists/numpy arrays directly (PyO3's `Vec<i32>`/`Vec<f64>` `FromPyObject` accepts a Python list or a 1-D numpy array of a compatible dtype natively, no capsule needed).
- No existing PyO3 codebase precedent for a polymorphic (float-or-list) return — resolved above by splitting `average=None` into a `_per_class` function.

### Files
- Create: `crates/mlrs-py/src/metrics.rs`
- Create: `crates/mlrs-py/tests/test_metrics.py`
- Modify: `crates/mlrs-py/src/lib.rs` (add `pub mod metrics;` + 13 `m.add_function(wrap_pyfunction!(...))` calls — 11 metrics + 2 extra `_per_class` variants for precision/recall/f1's `average=None`, i.e. 11 base + 3 `_per_class` = 14 total registrations; exact count confirmed at Green when every signature is finalized)
- Modify: `crates/mlrs-py/src/errors.rs` (add `metric_err_to_py`)

### TDD Sequence

#### 1. Red
- Test name: `test_accuracy_score_length_mismatch_raises_value_error` in `crates/mlrs-py/tests/test_metrics.py`.
- Setup: `_mlrs = pytest.importorskip("mlrs._mlrs")` (mirrors `test_naive_bayes.py:45`).
- Call `_mlrs.accuracy_score([1,0,1], [1,0], None, True)` (mismatched lengths).
- Expected: `pytest.raises(ValueError)`.
- Expected initial failure: `AttributeError: module 'mlrs._mlrs' has no attribute 'accuracy_score'` (the function is not yet registered).
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/tests/test_metrics.py::test_accuracy_score_length_mismatch_raises_value_error`

#### 2. Green
- Create `metrics.rs` with one `#[pyfunction]` per algos function (plain `Vec<i32>`/`Vec<f64>` params, matching SPEC §4's PyO3 skeleton), e.g.:
  ```rust
  #[pyfunction]
  #[pyo3(signature = (y_true, y_pred, sample_weight=None, normalize=true))]
  fn accuracy_score(y_true: Vec<i32>, y_pred: Vec<i32>, sample_weight: Option<Vec<f64>>, normalize: bool) -> PyResult<f64> {
      if y_true.len() != y_pred.len() {
          return Err(PyValueError::new_err("y_true and y_pred must be the same length"));
      }
      Ok(mlrs_algos::metrics::classification::accuracy_score(&y_true, &y_pred, sample_weight.as_deref(), normalize))
  }
  ```
  (length check here since `accuracy_score`'s algos-level signature is infallible `f64`, not `Result` — TASK-03's Green decision; the PyO3 layer is where the length-mismatch surfaces as `ValueError` for this particular metric). For `Result`-returning algos fns (`confusion_matrix` if TASK-04 made it fallible, `roc_auc_score_binary`/`_multiclass`), map via the new `metric_err_to_py`.
  - `average`/`multi_class`/`zero_division` cross the boundary as `&str`/`f64` per the resolved-decisions section (`average: &str` with `"none"` sentinel routed to a separate `_per_class` fn returning `Vec<f64>`; `zero_division: f64` where `f64::NAN` represents the `'nan'` policy).
- Add `pub mod metrics;` to `lib.rs`; register every function.
- Add `metric_err_to_py` to `errors.rs`.
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/tests/test_metrics.py::test_accuracy_score_length_mismatch_raises_value_error`

#### 3. Refactor
- Confirm every `#[pyfunction]` follows the SAME parameter-crossing convention (no ad-hoc divergence per metric) before adding the remaining smoke tests.
- Run: `cargo test -p mlrs-py --features cpu`

#### 4. Verify
- Run: `cargo build -p mlrs-py --features cpu` (compiles without `extension-module`, per `Cargo.toml:25-29` comment — this is the `cargo test` link mode, `[VERIFIED: LOCAL]`)
- Run: `cargo test -p mlrs-py --features cpu`
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/tests/test_metrics.py`
- Confirm: every one of the 11 metrics (14 registrations incl. `_per_class` variants) is callable from `_mlrs` and every length-mismatch/error-path smoke test passes.

### Implementation Steps
1. Red test #1 (above).
2. Add one smoke test per metric family (not per average/degenerate combination — that numeric-correctness gate already lives in TASK-03..14's Rust oracle tests; THIS layer only proves the binding surface is wired and error-mapped): `test_confusion_matrix_callable`, `test_precision_recall_f1_callable_and_per_class_variant`, `test_log_loss_callable`, `test_roc_auc_binary_single_class_raises_value_error`, `test_roc_auc_multiclass_callable`, `test_precision_recall_curve_callable`, `test_r2_mse_mae_callable`.
3. Implement `metrics.rs` with all 14 `#[pyfunction]`s in one Green pass, register in `lib.rs`, add `metric_err_to_py`.
4. Run the full Rust (`cargo test -p mlrs-py --features cpu`) and Python (`pytest crates/mlrs-py/tests/test_metrics.py`) suites.

### Completion Criteria
- [x] All Red tests fail (missing/unregistered function) before Green.
- [x] All pass after Green.
- [x] Every metric + its `_per_class` variant (where applicable) is registered in `_mlrs`.
- [x] `metric_err_to_py` exists in `errors.rs` alongside `algo_err_to_py`, same pattern.

### Risks and Guardrails
- Risk: forgetting a registration line in `lib.rs` (silent — Python `AttributeError` at call time, not a compile error) — the per-metric-family smoke test in step 2 is the guardrail (calling every function at least once).
- Risk: `average="none"`/`_per_class` split adds API surface sklearn's own `average=None` does not have at the Rust/PyO3 layer — this asymmetry MUST be hidden by the Python shim (TASK-16), which is exactly why `_mlrs.precision_score_per_class` should be treated as a low-level implementation detail, never sklearn-signature-faithful itself (only the shim is).

---

## TASK-16 — METR-SHIM-01: mlrs.metrics Python submodule

- **Spec:** `METR-SHIM-01`
- **Order:** 6 (Wave 5)
- **Depends on:** TASK-15

### Objective
`crates/mlrs-py/python/mlrs/metrics.py` exposes sklearn-signature-faithful
free functions (`accuracy_score`, `confusion_matrix`, `precision_score`,
`recall_score`, `f1_score`, `log_loss`, `roc_auc_score`,
`precision_recall_curve`, `r2_score`, `mean_squared_error`,
`mean_absolute_error`); `mlrs/__init__.py` gains `from . import metrics` (a
submodule import, NOT top-level `__all__` entries).

### Specification References
- `SPEC-METR-SHIM-01`

### Context and Evidence
- `crates/mlrs-py/python/mlrs/random_projection.py:44-66` (`johnson_lindenstrauss_min_dim` shim) — the exact lazy-`_load_ext()` + `np.asarray` + delegate-to-`_mlrs` pattern this module's functions follow (`[VERIFIED: CODEGRAPH]`).
- `crates/mlrs-py/python/mlrs/__init__.py:22-98,108-142` — `_load_ext()` (lazy import, clear `ImportError`) and the existing `__all__`/import-block convention; a SUBMODULE import (`from . import metrics`) is added separately from the top-level estimator re-exports (`[VERIFIED: CODEGRAPH]`) — do NOT add individual metric names to the top-level `__all__` list (SPEC §5 explicit instruction, avoids `mlrs.accuracy_score` colliding with estimator namespace).
- Free functions do NOT subclass `MlrsBase` — no `output_type`/`_normalize`/`_to_output` machinery applies (RESEARCH-METRICS §5, `[VERIFIED: LOCAL base.py:28-117 is estimator-fit/predict-oriented]`).
- SPEC §2 (revised, Plan-Check Issue 3): multioutput regression is a NON-GOAL. `r2_score`/`mean_squared_error`/`mean_absolute_error` MUST raise `NotImplementedError` on a 2-D `y_true`/`y_pred` or a non-default `multioutput=` kwarg, fail-closed — NEVER silently `ravel()` a 2-D array into a mathematically wrong 1-D value (`ravel`ing 2-D for r2 gives `1−ΣSSres/ΣSStot ≠ mean_k(1−SSres_k/SStot_k)`, SPEC §2 explicit derivation). This task's Red step below adds the dedicated fail-closed test.

### Files
- Create: `crates/mlrs-py/python/mlrs/metrics.py`
- Modify: `crates/mlrs-py/python/mlrs/__init__.py` (add `from . import metrics` submodule import; do NOT touch `__all__`)

### TDD Sequence

#### 1. Red
- Test name: `test_mlrs_metrics_submodule_importable_and_accuracy_score_matches_sklearn_signature` in a NEW file `crates/mlrs-py/python/tests/test_metrics_shim.py` (a Wave-5-scoped smoke test, distinct from TASK-17..23's oracle-replay tests which need TASK-16 to already exist).
- Setup: `import mlrs.metrics`.
- Call `mlrs.metrics.accuracy_score(np.array([1,0,1]), np.array([1,0,0]))`.
- Expected: `from mlrs.metrics import accuracy_score` succeeds; the call returns a plain Python `float` (`2/3`), matching sklearn's return type (a numpy scalar/float, not an array).
- Expected initial failure: `ModuleNotFoundError: No module named 'mlrs.metrics'` (the shim file does not exist yet).
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/python/tests/test_metrics_shim.py::test_mlrs_metrics_submodule_importable_and_accuracy_score_matches_sklearn_signature`

#### 2. Green
- Create `metrics.py`: each function `np.asarray(...).ravel()`-normalizes its inputs to the right dtype (`int32`/`int64` for labels, `float64` for scores/targets/proba), lazily resolves `_mlrs` via `from . import _load_ext` (mirroring `random_projection.py:58-60`), calls the corresponding `_mlrs.<fn>` (dispatching to the `_per_class` variant internally when `average is None`, per TASK-15's resolved API asymmetry — HIDDEN from the shim's own sklearn-faithful signature), and wraps the return: scalar → `float(...)`; `confusion_matrix` → `np.asarray(..., dtype=np.int64 if unweighted else np.float64)`; `precision_recall_curve` → tuple of `np.asarray(...)`; `average=None` → `np.asarray(...)`.
- `log_loss(..., eps='auto', ...)` maps the string `'auto'` to `1e-15` before calling `_mlrs.log_loss` (SPEC §4 Q5 resolution).
- `r2_score`/`mean_squared_error`/`mean_absolute_error` each check `np.asarray(y_true).ndim > 1 or np.asarray(y_pred).ndim > 1 or multioutput != 'uniform_average'` FIRST and raise `NotImplementedError("multioutput is not supported; pass 1-D y_true/y_pred")` BEFORE any `.ravel()` call (Plan-Check Issue 3 — fail-closed, never a silently-wrong `ravel()`ed value).
- Add `from . import metrics` to `__init__.py` (submodule import block, separate from the estimator re-export list at lines 22-61).
- Run: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/python/tests/test_metrics_shim.py`

#### 3. Refactor
- Confirm every shim function's kwarg names/defaults match sklearn's exactly (cross-check against SPEC §5's enumerated signatures) before moving to TASK-17's full oracle replay.
- Confirm the 2-D/`multioutput` check in all three regression shims is a SHARED helper (not three copy-pasted checks) so a future regression metric added to this file inherits the fail-closed behavior automatically.
- Run: `pytest crates/mlrs-py/python/tests/test_metrics_shim.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_metrics_shim.py`
- Confirm: `import mlrs; mlrs.metrics.r2_score` resolves (matches SPEC §5's "Users cannot call `mlrs.metrics.r2_score(...)`" — now they can).
- Confirm: estimator-enumerating gates remain unaffected — run `pytest crates/mlrs-py/python/tests/test_params.py crates/mlrs-py/python/tests/test_shims.py crates/mlrs-py/python/tests/test_estimator_checks.py` and confirm NO new failures (these are estimator-only, exempt per SPEC §5 METR-SHIM-01 acceptance note, `[VERIFIED: LOCAL test_params.py:12,53,255 keyed on estimator classes]`).

### Implementation Steps
1. Red test #1 (above, submodule import + signature).
2. Red test #2: `test_r2_score_2d_input_raises_not_implemented_error` (Plan-Check Issue 3) — `pytest.raises(NotImplementedError): mlrs.metrics.r2_score(np.zeros((3, 2)), np.zeros((3, 2)))`.
3. Red test #3: `test_mean_squared_error_non_default_multioutput_raises_not_implemented_error` — `pytest.raises(NotImplementedError): mlrs.metrics.mean_squared_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]), multioutput='raw_values')`.
4. Implement all 11 shim functions + the shared fail-closed multioutput guard in one Green pass (mechanical repetition of one pattern — a single spec, single principal failure reason: "the submodule and its functions exist and are sklearn-signature-faithful, and fail closed on the non-goal multioutput path").
5. Add the `__init__.py` submodule import.
6. Run the exemption regression check (test_params/test_shims/test_estimator_checks unaffected).

### Completion Criteria
- [x] All 3 Red tests fail (missing module / wrong behavior) before Green.
- [x] All 3 pass after Green.
- [x] `mlrs.metrics.<fn>` importable for all 11 sklearn-named functions.
- [x] A 2-D `y_true`/`y_pred` or non-default `multioutput=` on any of the three regression metrics raises `NotImplementedError`, never a silently-`ravel()`ed wrong value (Plan-Check Issue 3).
- [x] `test_params.py`/`test_shims.py`/`test_estimator_checks.py` show no new failures (metrics are exempt, not covered).

### Risks and Guardrails
- Risk: accidentally adding metric names to top-level `__all__` (would collide with the estimator namespace convention) — the Refactor step's signature cross-check + this task's explicit "do NOT touch `__all__`" constraint is the guardrail.
- Risk: implementing the 2-D check AFTER a `.ravel()` call (order bug) would silently defeat the fail-closed intent — Red tests #2/#3 are the guardrail; the Green step explicitly states the check must run FIRST.

---

## TASK-17 — Python oracle replay: accuracy_score + confusion_matrix

- **Spec:** `METR-CLS-01`, `METR-CLS-02`
- **Order:** 7 (Wave 6, first)
- **Depends on:** TASK-16, TASK-02

### Objective
`crates/mlrs-py/python/tests/test_oracle_metrics.py` replays the
`metrics_cls_binary_*`/`metrics_cls_multiclass_*`/`metrics_cls_degenerate`
fixtures through the FULL `mlrs.metrics` Python binding path for
`accuracy_score` and `confusion_matrix` (a SECOND consumer of the same
fixtures TASK-03/TASK-04 already gate at the Rust layer).

### Context and Evidence
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-81` — the exact oracle-replay-through-full-binding-path template (`_atol(fixture)` dtype-branch, `@requires_f64`, `conftest.dtype_of`/`fixture_path`) this file follows (`[VERIFIED: CODEGRAPH]`).
- `crates/mlrs-py/python/tests/conftest.py:35-49,146-152` — `fixture_path`/`dtype_of`/`requires_f64` helpers, reused verbatim (`[VERIFIED: LOCAL]`).

### Files
- Create: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_accuracy_score_oracle` (parametrized over `["metrics_cls_binary_f32_seed42", "metrics_cls_binary_f64_seed42"]`).
- Setup: `d = np.load(fixture_path(fixture))`; call `mlrs.metrics.accuracy_score(d["y_true"], d["y_pred"], sample_weight=d["sample_weight"])`.
- Expected: `abs(got - d["ref_accuracy_sw"][0]) <= _atol(fixture)`.
- Expected initial failure: `ModuleNotFoundError` if TASK-16 is incomplete, OR a numeric assertion failure if the shim mis-wires `sample_weight` — either way this is the first test in a NEW file, so its baseline failure mode is "file/collection succeeds but the specific test fails/errors before the shim is correct" (by this point in the plan TASK-16 already landed, so the Red state here specifically exercises the ORACLE REPLAY layer, distinguishing a Python-shim wiring bug from the already-passing Rust-layer correctness).
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_accuracy_score_oracle`

#### 2. Green
- If TASK-15/16 wired `sample_weight` correctly end-to-end, this test passes with NO new production code — it is a pure regression/integration confirmation. If it fails, the fix is confined to `metrics.py`'s `accuracy_score` shim (e.g. a missed `.ravel()` or dtype cast), NOT the algos layer (already gated green by TASK-03).
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_accuracy_score_oracle`

#### 3. Refactor
- None expected beyond matching the `test_oracle_neighbors.py` file structure/imports.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: `confusion_matrix` oracle (`ref_confusion`/`ref_confusion_sw`) + the empty-class/all-one-class degenerate cases (`ref_confusion_empty` via explicit `labels=[0,1,2]`, `ref_confusion_one`) all pass through the full binding path.

### Implementation Steps
1. Red test #1 (above, `accuracy_score` weighted oracle).
2. Red test #2: `test_accuracy_score_single_sample_degenerate` (`ref_acc_single_match`/`ref_acc_single_mismatch`).
3. Red test #3: `test_confusion_matrix_oracle` (binary + multiclass, `ref_confusion`/`ref_confusion_sw`).
4. Red test #4: `test_confusion_matrix_empty_class_via_labels` (`labels=[0,1,2]`, `ref_confusion_empty`).
5. Red test #5: `test_confusion_matrix_all_one_class` (`ref_confusion_one`).
6. Green: fix any shim wiring gaps surfaced.

### Completion Criteria
- [x] All 5 tests pass through the full `numpy -> mlrs.metrics -> _mlrs -> Rust` path.
- [x] `@requires_f64` gates the f64 parametrization per fixture (mirrors `test_oracle_neighbors.py:30-31`).

### Risks and Guardrails
- Risk: none beyond standard shim-wiring gaps — this task's tests are a regression net on TASK-15/16, not new algorithmic risk.

---

## TASK-18 — Python oracle replay: precision/recall/f1 (all averages + zero_division)

- **Spec:** `METR-CLS-03`, `METR-CLS-04`, `METR-CLS-05`
- **Order:** 8 (Wave 6)
- **Depends on:** TASK-17

### Objective
Extend `test_oracle_metrics.py` to replay every `average` value + the
per-metric `zero_division` degenerate for `precision_score`/`recall_score`/
`f1_score` through the full binding path, including the `average=None`
per-class-vector return type.

### Context and Evidence
- Same template as TASK-17 (`test_oracle_neighbors.py` pattern).
- Fixtures: `metrics_cls_multiclass_*_seed42.npz` (`ref_precision_{macro,micro,weighted,none}`, ditto recall/f1, plus `ref_precision_labelreorder`/`ref_recall_labelreorder`/`ref_f1_labelreorder` + `y_true_labelreorder`/`y_pred_labelreorder`/`labels_reorder` — Plan-Check Issue 6), `metrics_cls_degenerate_seed42.npz` (`ref_precision_zerodiv`, `ref_recall_zerodiv`, `ref_f1_zerodiv`).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_precision_score_average_none_returns_per_class_array`.
- Call `mlrs.metrics.precision_score(d["y_true"], d["y_pred"], average=None)`.
- Expected: a numpy array (NOT a scalar) matching `ref_precision_none` elementwise — this specifically exercises the `average=None` → `_per_class` dispatch asymmetry TASK-15/16 introduced (the riskiest wiring point in the shim).
- Expected initial failure: shim wiring bug (wrong dispatch, wrong shape, or a raised exception) if `average=None` routing is incomplete; otherwise a clean pass confirming the earlier Green already covers it.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_precision_score_average_none_returns_per_class_array`

#### 2. Green
- Fix any `average=None` dispatch gap in `metrics.py` (route to `_mlrs.precision_score_per_class` and return `np.asarray(...)`).
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_precision_score_average_none_returns_per_class_array`

#### 3. Refactor
- Mirror the same `average=None` test for `recall_score`/`f1_score` once the pattern is confirmed correct for `precision_score`.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: every `average ∈ {binary,macro,micro,weighted,None}` for all three metrics, plus each metric's own zero-division degenerate, passes through the full path.

### Implementation Steps
1. Red test #1 (above, precision `average=None`).
2. Red tests #2-#3: same `average=None` check for `recall_score`/`f1_score`.
3. Red tests #4-#6: `test_{precision,recall,f1}_score_zero_division_degenerate` (`ref_{precision,recall,f1}_zerodiv`).
4. Red tests #7-#9: `test_{precision,recall,f1}_score_averages_oracle` (parametrized over `{binary,macro,micro,weighted}` against the multiclass/binary fixtures).
5. Red tests #10-#12: `test_{precision,recall,f1}_score_labels_reorder_oracle` (Plan-Check Issue 6) — `mlrs.metrics.{precision,recall,f1}_score(d["y_true_labelreorder"], d["y_pred_labelreorder"], labels=d["labels_reorder"], average='macro')` against `ref_{precision,recall,f1}_labelreorder`, proving the `labels` kwarg reorders classes through the FULL Python binding path (not just at the Rust layer, TASK-05/06/07).
6. Green: fix wiring gaps found.

### Completion Criteria
- [x] All 12 tests pass through the full binding path.
- [x] `average=None` returns an array-like (not a scalar) for all three metrics.
- [x] The `labels`-reorder replay tests (Issue 6) pass for all three metrics through the full `mlrs.metrics` shim.

### Risks and Guardrails
- Risk (the highest-value check in this task): the `average=None` → `_per_class` PyO3-function-split (TASK-15's resolved-decision) is the ONE piece of shim logic with no earlier oracle-level test — this task is where it is FIRST exercised end-to-end.
- Risk: the shim's `labels` kwarg must be forwarded as an actual list/array to `_mlrs.<fn>`, not silently dropped — a dropped `labels` kwarg would still "work" (no exception) but silently ignore the reorder, so the labels-reorder tests assert the VALUE, not just the absence of an exception.

---

## TASK-19 — Python oracle replay: log_loss

- **Spec:** `METR-CLS-06`
- **Order:** 9 (Wave 6)
- **Depends on:** TASK-18

### Objective
Extend `test_oracle_metrics.py` to replay `log_loss` (multiclass oracle +
weighted + the `0.0`/`1.0` clipping degenerate + `eps='auto'` string mapping)
through the full binding path.

### Context and Evidence
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_log_loss_binary`/`y_prob_binary` — Plan-Check Issue 7), `metrics_cls_multiclass_*_seed42.npz` (`ref_log_loss`, `ref_log_loss_sw`), `metrics_cls_degenerate_seed42.npz` (`y_true_clip`/`y_prob_clip`/`ref_log_loss_clip`, `y_true_logloss_labelreorder`/`y_prob_logloss_labelreorder`/`labels_logloss_reorder`/`ref_log_loss_labelreorder` — Plan-Check Issue 6).
- SPEC §4 Q5: shim maps `eps='auto'` → `1e-15`.
- SPEC §2: "`labels` parameter for ... log_loss ... — each with its own reorder acceptance test" — this task's Python replay closes that acceptance clause end-to-end (TASK-08 already closes it at the Rust layer).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_log_loss_eps_auto_maps_to_fixed_epsilon`.
- Call `mlrs.metrics.log_loss(d["y_true_clip"], d["y_prob_clip"], eps='auto')`.
- Expected: `abs(got - ref_log_loss_clip) <= 1e-5` (a finite value — proves the `'auto'`→`1e-15` string mapping works end-to-end, not just at the Rust layer where `eps` is always numeric).
- Expected initial failure: `TypeError`/wrong-value if the shim passes the LITERAL STRING `'auto'` through to `_mlrs.log_loss` (which expects `f64`) instead of mapping it first.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_log_loss_eps_auto_maps_to_fixed_epsilon`

#### 2. Green
- Fix `metrics.py::log_loss` to map `eps='auto'` → `1e-15` before calling `_mlrs.log_loss` (if not already done correctly in TASK-16).
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_log_loss_eps_auto_maps_to_fixed_epsilon`

#### 3. Refactor
- None expected.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: multiclass oracle + weighted variants pass.

### Implementation Steps
1. Red test #1 (above, `eps='auto'` mapping).
2. Red test #2: `test_log_loss_matches_sklearn_oracle_multiclass` (`ref_log_loss`, multiclass, ≤1e-5).
3. Red test #3: `test_log_loss_weighted_matches_sklearn_oracle` (`ref_log_loss_sw`).
4. Red test #4: `test_log_loss_matches_sklearn_oracle_binary` (Plan-Check Issue 7) — `mlrs.metrics.log_loss(d["y_true"], d["y_prob_binary"])` against `ref_log_loss_binary`, closing the explicit binary acceptance clause end-to-end.
5. Red test #5: `test_log_loss_labels_reorder_matches_sklearn_oracle` (Plan-Check Issue 6) — `mlrs.metrics.log_loss(d["y_true_logloss_labelreorder"], d["y_prob_logloss_labelreorder"], labels=d["labels_logloss_reorder"])` against `ref_log_loss_labelreorder`.
6. Green: fix any gap.

### Completion Criteria
- [x] All 5 tests pass through the full binding path.
- [x] `eps='auto'` is confirmed mapped correctly end-to-end (not just at the Rust unit-test layer).
- [x] The explicit binary `log_loss` (Issue 7) and `labels`-reorder (Issue 6) replay tests both pass.

### Risks and Guardrails
- Risk: `eps='auto'` is a Python-string-only concept (the Rust layer never sees it) — this is the ONE place that mapping is testable; do not skip it.
- Risk: the `labels` kwarg silently dropped by the shim (same risk as TASK-18) — the labels-reorder test asserts the VALUE against `ref_log_loss_labelreorder`, not merely the absence of an exception.

---

## TASK-20 — Python oracle replay: roc_auc_score (binary)

- **Spec:** `METR-CLS-07`
- **Order:** 10 (Wave 6)
- **Depends on:** TASK-19

### Objective
Extend `test_oracle_metrics.py` to replay binary `roc_auc_score` (tie-heavy
oracle + weighted + the single-class `ValueError` gate) through the full
binding path.

### Context and Evidence
- Fixtures: `metrics_cls_binary_*_seed42.npz` (`ref_roc_auc`, `ref_roc_auc_sw`), `metrics_cls_degenerate_seed42.npz` (`y_true_singleclass`/`y_score_singleclass`).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_roc_auc_score_binary_single_class_raises_value_error`.
- Call `mlrs.metrics.roc_auc_score(d["y_true_singleclass"], d["y_score_singleclass"])` inside `pytest.raises(ValueError)`.
- Expected: `ValueError` raised (matching sklearn's own single-class behavior — SPEC §6).
- Expected initial failure: any deviation (no error raised, or a different exception type) if TASK-15's `metric_err_to_py` mapping or TASK-16's shim swallows/mistranslates the error.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_roc_auc_score_binary_single_class_raises_value_error`

#### 2. Green
- Fix any error-propagation gap (should already work if TASK-15's `Result<f64, MetricError>` → `PyValueError` mapping is correctly wired).
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_roc_auc_score_binary_single_class_raises_value_error`

#### 3. Refactor
- None expected.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: tie-heavy oracle + weighted variants pass.

### Implementation Steps
1. Red test #1 (above, single-class error gate — the load-bearing check for this task).
2. Red test #2: `test_roc_auc_score_binary_matches_sklearn_oracle` (`ref_roc_auc`, tie-heavy, ≤1e-5).
3. Red test #3: `test_roc_auc_score_binary_weighted_matches_sklearn_oracle` (`ref_roc_auc_sw`).
4. Green: fix any gap.

### Completion Criteria
- [x] All 3 tests pass through the full binding path.
- [x] The single-class case raises `ValueError`, matching sklearn's behavior class (not a different exception).

### Risks and Guardrails
- Risk: an error silently swallowed into a `NaN` return (instead of propagating) would be a serious correctness regression — the `pytest.raises(ValueError)` assertion is the hard gate against this.

---

## TASK-21 — Python oracle replay: roc_auc_score (multiclass OvR/OvO)

- **Spec:** `METR-CLS-08`
- **Order:** 11 (Wave 6)
- **Depends on:** TASK-20

### Objective
Extend `test_oracle_metrics.py` to replay all 4 unweighted
`(multi_class, average)` combinations for multiclass `roc_auc_score`, the 2
weighted-OvR combinations (Plan-Check Issue 1), and the OvO+`sample_weight`
carve-out (Plan-Check Issue 2 — either a value match or a `ValueError` gate,
mirroring TASK-10's exact branch), through the full binding path.

### Context and Evidence
- Fixture: `metrics_cls_multiclass_*_seed42.npz` (`ref_roc_auc_ovr_macro`, `ref_roc_auc_ovr_weighted`, `ref_roc_auc_ovo_macro`, `ref_roc_auc_ovo_weighted`, `ref_roc_auc_ovr_macro_sw`, `ref_roc_auc_ovr_weighted_sw` — Issue 1, always present; `ref_roc_auc_ovo_macro_sw`/`ref_roc_auc_ovo_weighted_sw` ONLY if TASK-02's probe found Branch B).
- Mirrors TASK-10's exact two-branch OvO+`sample_weight` carve-out — read TASK-02's recorded probe outcome (same one TASK-10 used) before writing this task's weighted-OvO test.

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_roc_auc_score_multiclass_ovr_macro_oracle` (parametrized to cover all 4 unweighted combos as separate parametrize cases: `[("ovr","macro"),("ovr","weighted"),("ovo","macro"),("ovo","weighted")]`).
- Call `mlrs.metrics.roc_auc_score(d["y_true"], d["y_proba"], multi_class=mc, average=avg)`.
- Expected: matches the corresponding `ref_roc_auc_{mc}_{avg}` within `1e-5`.
- Expected initial failure: shim kwarg-wiring gap (e.g. `multi_class`/`average` not forwarded to the correct `_mlrs` function/enum string) if present; otherwise a clean pass.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_roc_auc_score_multiclass_ovr_macro_oracle`

#### 2. Green
- Fix any shim kwarg-forwarding gap.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py -k multiclass_ovr_macro`

#### 3. Refactor
- None expected beyond confirming all parametrize cases share one test body.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: all 4 unweighted combinations, both weighted-OvR combinations, and the OvO+weight carve-out (whichever branch applies) pass.

### Implementation Steps
1. Red test (parametrized, above) covering all 4 unweighted combinations.
2. Red test #2: `test_roc_auc_score_multiclass_ovr_sample_weight_oracle` (Plan-Check Issue 1) — parametrized over `average ∈ {macro, weighted}`, calling `mlrs.metrics.roc_auc_score(d["y_true"], d["y_proba"], multi_class='ovr', average=avg, sample_weight=d["sample_weight"])` against `ref_roc_auc_ovr_{macro,weighted}_sw`.
3. Red test #3: `test_roc_auc_score_multiclass_ovo_sample_weight_gate` (Plan-Check Issue 2) — EITHER `pytest.raises(ValueError): mlrs.metrics.roc_auc_score(d["y_true"], d["y_proba"], multi_class='ovo', average='macro', sample_weight=d["sample_weight"])` (Branch A, mirrors TASK-10's `Err(MetricError::WeightedOvoUnsupported)` mapped through `metric_err_to_py` to `PyValueError`) OR a value assertion against `ref_roc_auc_ovo_macro_sw` (Branch B) — write whichever branch TASK-02/TASK-10 used, the SAME branch, not independently re-decided.
4. Green: fix any gap.

### Completion Criteria
- [x] All 4 unweighted parametrized cases pass through the full binding path.
- [x] The 2 weighted-OvR cases pass (Issue 1).
- [x] The OvO+weight carve-out test passes, using the SAME branch (error gate or value) that TASK-10 implemented at the Rust layer — a mismatch here (e.g. Python expects a value but Rust raises) would itself be a real bug this test is designed to catch.

### Risks and Guardrails
- Risk: same OvO pairwise-averaging risk as TASK-10, now re-verified end-to-end through the Python shim (a second, independent consumer of the same Rust value).
- Risk: this task's OvO+weight test branch must match TASK-10's branch EXACTLY — if TASK-10 implemented the error gate but this task asserts a value (or vice versa), that is a genuine cross-layer inconsistency bug, not a flaky test; investigate the mismatch rather than "fixing" the test to whichever branch happens to pass.

---

## TASK-22 — Python oracle replay: precision_recall_curve

- **Spec:** `METR-CLS-09`
- **Order:** 12 (Wave 6)
- **Depends on:** TASK-21

### Objective
Extend `test_oracle_metrics.py` to replay `precision_recall_curve` (trivial +
tie-heavy fixtures, sentinel/length invariants) through the full binding path.

### Context and Evidence
- Fixture: `metrics_cls_binary_*_seed42.npz` (`ref_pr_precision`, `ref_pr_recall`, `ref_pr_thresholds`, `ref_pr_precision_sw`, `ref_pr_recall_sw`, `ref_pr_thresholds_sw` — Plan-Check Issue 1, ALWAYS generated, so the weighted replay test below is REQUIRED, not conditional).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_precision_recall_curve_returns_three_arrays_with_sentinel`.
- Call `precision, recall, thresholds = mlrs.metrics.precision_recall_curve(d["y_true"], d["y_score"])`.
- Expected: `len(precision) == len(thresholds) + 1`, `precision[-1] == 1.0`, `recall[-1] == 0.0` — the SAME structural invariants TASK-11 gates at the Rust layer, now confirmed through the tuple-of-arrays Python return.
- Expected initial failure: a tuple/shape mismatch if the shim's tuple-wrapping (`np.asarray` per element) is wrong.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_precision_recall_curve_returns_three_arrays_with_sentinel`

#### 2. Green
- Fix any tuple-wrapping gap in `metrics.py::precision_recall_curve`.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_precision_recall_curve_returns_three_arrays_with_sentinel`

#### 3. Refactor
- None expected.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`
- Confirm: elementwise match against `ref_pr_precision`/`ref_pr_recall`/`ref_pr_thresholds` at ≤1e-5, AND the weighted variant against `ref_pr_precision_sw`/`ref_pr_recall_sw`/`ref_pr_thresholds_sw`.

### Implementation Steps
1. Red test #1 (above, structural).
2. Red test #2: `test_precision_recall_curve_matches_sklearn_oracle` (elementwise, tie-heavy).
3. Red test #3: `test_precision_recall_curve_weighted_matches_sklearn_oracle` (Plan-Check Issue 1 — REQUIRED) — `mlrs.metrics.precision_recall_curve(d["y_true"], d["y_score"], sample_weight=d["sample_weight"])` against `ref_pr_precision_sw`/`ref_pr_recall_sw`/`ref_pr_thresholds_sw`.
4. Green: fix any gap.

### Completion Criteria
- [x] All 3 tests pass through the full binding path.
- [x] The weighted `precision_recall_curve` replay test (Issue 1) passes, closing the locked `sample_weight` requirement end-to-end.

### Risks and Guardrails
- Risk: none beyond the standard tuple-wrapping shim risk already covered by the structural test.
- Risk: `sample_weight` silently dropped by the shim before reaching `_mlrs.precision_recall_curve` — the weighted test asserts VALUES against `ref_pr_*_sw`, which would differ measurably from the unweighted reference if the weight were dropped.

---

## TASK-23 — Python oracle replay: r2_score / mean_squared_error / mean_absolute_error

- **Spec:** `METR-REG-01`, `METR-REG-02`, `METR-REG-03`
- **Order:** 13 (Wave 6, last)
- **Depends on:** TASK-22

### Objective
Extend `test_oracle_metrics.py` to replay all three regression metrics
(standard oracle + weighted + constant-target r2 + perfect-prediction) through
the full binding path — the FINAL task in the plan, closing out every spec ID.

### Context and Evidence
- Fixture: `metrics_reg_{f32,f64}_seed42.npz` (`ref_r2`, `ref_r2_sw`, `ref_mse`, `ref_mse_sw`, `ref_mae`, `ref_mae_sw`, `ref_r2_const`, `ref_r2_perfect`, `ref_mse_perfect`, `ref_mae_perfect`).
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py:23-24` — `_atol(fixture)` dtype-branch convention, reused for the f32/f64 parametrization here too (`[VERIFIED: LOCAL]`).

### Files
- Modify: `crates/mlrs-py/python/tests/test_oracle_metrics.py`

### TDD Sequence

#### 1. Red
- Test name: `test_r2_score_constant_target_oracle`.
- Call `mlrs.metrics.r2_score(d["y_true_const"], d["y_pred_const"])`.
- Expected: matches `ref_r2_const` within `1e-5` (the SAME fixture-pinned value TASK-12 already gates at the Rust layer — now confirmed through the full Python path, closing the last untested layer of that degenerate case).
- Expected initial failure: shim dtype/ravel gap if present; otherwise clean pass.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_r2_score_constant_target_oracle`

#### 2. Green
- Fix any gap in `metrics.py::r2_score`.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py::test_r2_score_constant_target_oracle`

#### 3. Refactor
- None expected.
- Run: `pytest crates/mlrs-py/python/tests/test_oracle_metrics.py`

#### 4. Verify
- Run the FULL plan regression gate:
  - `cargo test -p mlrs-algos --features cpu`
  - `cargo test -p mlrs-algos --features wgpu`
  - `cargo test -p mlrs-py --features cpu`
  - `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml && pytest crates/mlrs-py/python/tests/ crates/mlrs-py/tests/test_metrics.py`
- Confirm: every metric, every degenerate case, every `average`/`multi_class` combination, and the estimator-enumerating exemption gates (`test_params.py`/`test_shims.py`/`test_estimator_checks.py`) all pass with no new failures.

### Implementation Steps
1. Red test #1 (above, constant-target r2).
2. Red test #2: `test_r2_score_perfect_prediction_oracle` (`ref_r2_perfect`).
3. Red test #3: `test_mean_squared_error_oracle` (standard + weighted + perfect).
4. Red test #4: `test_mean_absolute_error_oracle` (standard + weighted + perfect).
5. Green: fix any gaps.
6. Run the full-plan regression gate (all four commands above).

### Completion Criteria
- [x] All 4 tests pass through the full binding path.
- [x] The full-plan regression gate (all four validation commands) is green.
- [x] Every one of the 16 spec IDs (`METR-INFRA-01`, `METR-CLS-01..09`, `METR-REG-01..03`, `METR-BIND-01`, `METR-SHIM-01`, `METR-ORACLE-01`) has at least one passing test at both the Rust algos layer AND (where applicable per SPEC §6's acceptance matrix) the Python oracle-replay layer.

### Risks and Guardrails
- Risk: this is the last task — if the full-plan regression gate reveals a cross-task interaction bug (e.g. TASK-03's `nb_common` delegation regressed something TASK-15 depends on), it must be triaged back to the OWNING task's file, not patched ad hoc here.

---

## Specification → Task Coverage Map

| Spec ID | Task(s) |
|---|---|
| `METR-INFRA-01` | TASK-01 |
| `METR-CLS-01` | TASK-03, TASK-17 |
| `METR-CLS-02` | TASK-04, TASK-17 |
| `METR-CLS-03` | TASK-05, TASK-18 |
| `METR-CLS-04` | TASK-06, TASK-18 |
| `METR-CLS-05` | TASK-07, TASK-18 |
| `METR-CLS-06` | TASK-08, TASK-19 |
| `METR-CLS-07` | TASK-09, TASK-20 |
| `METR-CLS-08` | TASK-10, TASK-21 |
| `METR-CLS-09` | TASK-11, TASK-22 |
| `METR-REG-01` | TASK-12, TASK-23 |
| `METR-REG-02` | TASK-13, TASK-23 |
| `METR-REG-03` | TASK-14, TASK-23 |
| `METR-BIND-01` | TASK-15 |
| `METR-SHIM-01` | TASK-16 |
| `METR-ORACLE-01` | TASK-02 |

All 16 spec IDs in `SPEC.md` (revision 2) §5 are covered by at least one
task; every task cites at least one spec ID. Every locked/in-scope
acceptance clause carries either a Red test or a documented gate:
`sample_weight` on `precision_recall_curve` (TASK-11/22) and OvR `roc_auc`
(TASK-10/21) is now tested (Plan-Check Issue 1); the OvO+`sample_weight`
carve-out (Issue 2) is a documented two-branch gate in TASK-02/10/21; the
multioutput non-goal (Issue 3) is a fail-closed `NotImplementedError` Red
test in TASK-16; `metrics/mod.rs` is edited exactly once, in TASK-01 (Issue
4); every fixture array is float-cast, stated as an explicit TASK-02
completion criterion (Issue 5); `labels`-reorder is tested independently for
precision/recall/f1 (TASK-05/06/07 + TASK-18) and log_loss (TASK-08 +
TASK-19) (Issue 6); binary `log_loss` has its own oracle test (TASK-08 +
TASK-19, Issue 7); and the `nb_common` empty-input `NaN` contract has a
dedicated regression assertion (TASK-03, Issue 8).

## Unresolved Blockers / Unverified Assumptions Carried Into Implementation

1. **`log_loss` clip-vs-renormalize behavior** (TASK-08) — `[UNVERIFIED]`
   against the Planner's knowledge of the exact `scikit-learn==1.9.0`
   internals; TASK-02's Green step captures the ACTUAL fixture value and
   TASK-08's Green step resolves the implementation empirically against it,
   documenting the resolution in a doc-comment. Not a planning blocker (the
   fixture is the source of truth), but flagged so the implementer does not
   skip the empirical check.
2. **Confusion-matrix narrowing** (TASK-04) — whether sklearn silently drops
   rows whose true/pred label falls OUTSIDE an explicit `labels` narrower than
   the data is untested by the committed fixture design (which only exercises
   `labels` as a SUPERSET). Marked as an explicit non-goal unless a future
   fixture exercises it.
3. **Q6 (sklearn fixture version)** — resolved to `scikit-learn==1.9.0` based
   on the most recent in-repo precedent (`scripts/gen_oracle.py:931,3696`),
   since no sklearn is installed in this planning environment to query
   directly. If the regen operator's environment differs, TASK-02 must be
   rerun against the operator's actual pinned version and the docstring
   updated to match.
4. **PyO3 0.28.3 exact registration count** (TASK-15) — the plan estimates 14
   `#[pyfunction]` registrations (11 metrics + 3 `_per_class` variants for
   precision/recall/f1); confirmed exactly at Green once every signature is
   finalized (`confusion_matrix`/`log_loss`/`roc_auc_score_multiclass` do not
   need a `_per_class` split).
5. **Q10 (OvO + `sample_weight` support in `scikit-learn==1.9.0`)** —
   `[UNVERIFIED]` until TASK-02's Green step actually runs the probe; TASK-02,
   TASK-10, and TASK-21 all branch on the SAME recorded outcome (documented in
   the multiclass generator's docstring), so this is a single point of truth,
   not three independent guesses. Not a planning blocker — the probe result
   determines which of the two fully-specified branches is implemented.

## Confirmation

No GSD skill, command, workflow, or agent was invoked at any point in this
revision pass. Only `.planning/plans/metrics-surface/PLAN.md` was edited (in
place, preserving all 23 task numbers and their execution order). `SPEC.md`
was re-read in full (its `spec_revision: 2` changes — multioutput as a
non-goal, the OvO+`sample_weight` carve-out, the `metric_err_to_py` naming
correction, 1-D-only regression metrics, the new `MetricError::WeightedOvoUnsupported`
variant — were the coordinator's own prior edit, not made by this pass) but
not modified by this agent. `SOURCES.md`, `RESEARCH-METRICS.md`,
`.planning/STATE.md`, `.planning/ROADMAP.md`, and all production/test/fixture
files were read-only inspected via CodeGraph/Read/Bash, never modified.
