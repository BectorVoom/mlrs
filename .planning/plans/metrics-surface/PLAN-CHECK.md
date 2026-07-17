# Plan Check — mlrs Metrics Surface

**Agent:** plan-checker (independent, CodeGraph-verified). **Date:** 2026-07-16.

## Pass 1 — Verdict: ISSUES_FOUND

**Goal:** sklearn metrics surface — 11 metrics (classification + regression) host-only Rust in `crates/mlrs-algos/src/metrics/`, PyO3 free-function bindings, `mlrs.metrics` submodule, ≤1e-5 vs sklearn, `sample_weight` on every metric, `average∈{binary,macro,micro,weighted,None}`, mandatory degenerate fixtures.
**Plan reviewed:** `PLAN.md` (23 tasks).

### Verified-safe seams (CodeGraph)
- `nb_common::accuracy_score` (`crates/mlrs-algos/src/naive_bayes/nb_common.rs:160`): signature `(pred, y_true)` — opposite sklearn's arg order (plan flags this). **Exactly 1 caller**; covered by `nb_common_test.rs`. Function is swap-symmetric, so the delegation seam is SAFE and no caller edit is needed. Returns `f64::NAN` on empty (`:172-174`), panics on length mismatch.
- PyO3 free-fn pattern real: `johnson_lindenstrauss_min_dim` (`projection.rs:379-382`), `backend_supports_f64` (`lib.rs:167`), registered at `lib.rs:196,238`.
- `algo_err_to_py` (`errors.rs:56`) maps `AlgoError`→`PyValueError`; new `metric_err_to_py` sibling correctly planned for the distinct `MetricError`.
- Oracle template real: `load_npz`/`OracleCase`/`expect_f64` (`crates/mlrs-core/src/oracle.rs:35,60,77`), `skip_f64_with_log` (`crates/mlrs-backend/src/capability.rs:147`).
- `float_dtype` (`ingress.rs:112-118`) is float-only — validates plain-`Vec` ingress for integer labels.
- `_FIXTURE_DIR` (`gen_oracle.py:41`), `main()` dispatch convention confirmed.

### Issues (severity → resolution)
1. **[CRITICAL] sample_weight unverified for pr_curve + multiclass roc_auc.** Weighted code paths ship un-compared to sklearn → silent ≤1e-5 violation of a locked requirement. → Add weighted fixtures + Red tests (TASK-02, TASK-10/21, TASK-11/22).
2. **[MAJOR] sklearn OvO roc_auc may reject sample_weight.** Cannot verify (sklearn not installed). → TASK-02 probes; if unsupported, SPEC carves OvO out of the sample_weight requirement and Rust OvO rejects `sw!=None` to match sklearn.
3. **[MAJOR] `multioutput='uniform_average'` in-scope + in acceptance but no task/signature/fixture.** `ravel()` on 2-D y gives numerically wrong r2. → Downgrade multioutput to NON-GOAL; shim raises `NotImplementedError` for 2-D y / non-default `multioutput`.
4. **[MAJOR] Wave 3a/3b both edit `metrics/mod.rs` — "parallel-eligible" is invalid.** Concurrent edits conflict / intermediate build breaks. → TASK-01 pre-creates `classification.rs`+`regression.rs` stubs and both `pub mod` lines up front.
5. **[MAJOR] `load_npz` rejects non-float arrays.** Integer label/count fixtures fail to load entirely (`oracle.rs:115-135` accepts only 4/8-byte floats). → TASK-02 must cast EVERY saved array (labels, `labels_*`, all `ref_confusion*`) to float32/64; add a completion checkbox.
6. **[MAJOR] `labels` parameter untested for P/R/F1 + log_loss** (in-scope, named acceptance clause). → Add `labels`-reorder Red tests to TASK-07/08 + TASK-18/19 + refs in TASK-02.
7. **[MINOR] Binary log_loss fixture missing.** → Add binary ref, or note multiclass subsumes binary.
8. **[MINOR] nb_common empty→NaN contract untested.** → Add empty-input NaN assertion to TASK-03.

### Order review (valid except)
- 3a/3b `mod.rs` collision (Issue 4). Otherwise: fixtures→loaders, algos→bind→shim→python-replay, roc_auc sweep helper before multiclass/pr_curve — all valid.

### Unverified at plan time (defer to TASK-02 Green against the real env)
- sklearn OvO+sample_weight behavior; exact sklearn version (`scikit-learn==1.9.0` sourced from `gen_oracle.py:931,3696`); log_loss clip-vs-renormalize (fixture is source of truth); final `#[pyfunction]` count (~14).

---

## Pass 2 — Verdict: PASS

Re-reviewed revised `PLAN.md` (23 tasks) + `SPEC.md` (spec_revision: 2). **All 8 pass-1 issues fully resolved; no new blocker/major defect introduced.**

CodeGraph re-verification of the two load-bearing claims:
- `crates/mlrs-core/src/oracle.rs:97-146` — `read_named_arrays` matches `num_bytes` only against `Some(8)`/`Some(4)` floats; any other dtype aborts the whole file load. Confirms Issue 5 root cause; TASK-02's cast-every-array rule is necessary and correctly scoped (incl. the degenerate generator → `np.float64`).
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs:160-181` — `accuracy_score(pred, y_true)` (arg order opposite sklearn), empty→`NaN` (`:172-174`), 1 caller. Validates TASK-03's delegation seam + the Issue-8 NaN regression assertion (`0.0/0.0` preserves the contract).

Resolution ledger: (1) weighted pr_curve/OvR-roc_auc now REQUIRED Red tests + `ref_*_sw` fixtures, hedge removed; (2) OvO+sample_weight probe in TASK-02 drives BOTH the fixture and the Rust branch off one recorded outcome (no divergence) — `MetricError::WeightedOvoUnsupported` + SPEC §2 carve-out; (3) multioutput non-goal, TASK-16 `NotImplementedError` guards before `.ravel()`, no residual wording; (4) `metrics/mod.rs` edited exactly once (TASK-01 stubs both submodules) — Wave 3a∥3b now genuinely disjoint; (5) float-cast rule + checkbox + per-array verify; (6) labels-reorder Red tests for P/R/F1 + log_loss (Rust + Python); (7) binary log_loss ref; (8) empty→NaN assertion.

New-defect scan clean: no dangling fixture references (single-class roc_auc appears only as a negative guardrail), stub files compile empty, TASK-10/21 documentation-mediated on TASK-02's probe with wave ordering guaranteeing freshness, all 16 spec IDs still covered.

Non-blocking minor notes (implementer flags, not gate failures):
- Add an explicit "Depends on: TASK-02 (recorded probe outcome)" edge to TASK-10/21 for self-documentation (dependency is transitively satisfied by wave order).
- TASK-01 pins the shared-bookkeeping API at plan time; if a later task needs more, the plan's guardrail adds a NEW task rather than reopening `mod.rs` — the one spot where the "edited once" invariant could be pressured.

Deferred runtime facts (correctly routed to TASK-02 Green against the real regen env; fixture is source of truth, no code path silently assumes an outcome): sklearn OvO+sample_weight behavior, exact pinned sklearn version (`scikit-learn==1.9.0` per `gen_oracle.py:931,3696`), log_loss clip-vs-renormalize.

**Gate satisfied: independent Plan Checker PASS on pass 2. Plan is ready for implementation.**
