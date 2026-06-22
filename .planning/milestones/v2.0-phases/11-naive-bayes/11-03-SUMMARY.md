---
phase: 11-naive-bayes
plan: 03
subsystem: algos
tags: [naive-bayes, multinomial-nb, bernoulli-nb, complement-nb, gemm, log-sum-exp, oracle, sklearn, NB-02, NB-03, NB-04]

# Dependency graph
requires:
  - phase: 11-naive-bayes
    plan: 01
    provides: "nb_common GATHER (class_grouped_sum) + log_sum_exp_normalize + empirical_class_log_prior + argmax/argmin_decode; PredictLogProba trait; the three discrete Wave-0 stubs (builder + build() validation incl. D-06 force_alpha clip + geometry guard); committed multinomial/bernoulli/complement_nb_{f32,f64}_seed42 fixtures; validate_discrete_alpha shared free fn"
  - phase: 11-naive-bayes
    plan: 02
    provides: "GaussianNB fit/predict pattern (decode classes, GATHER sufficient stats, host log-sum-exp through the three predict traits, WR-07 release-on-refit) — the structural template the three count variants follow"
  - phase: 02-gemm
    provides: "gemm prim (transb=true reads the (n_classes, n_features) buffer as its transpose) — the shared device joint-LL matvec"
provides:
  - "MultinomialNB<F>: Fit + PredictLabels + PredictProba + PredictLogProba (NB-02) — feature_log_prob_ with the alpha·n_features denominator, GEMM joint-LL, exact sklearn labels (f32+f64)"
  - "BernoulliNB<F> (NB-03): binarize Option<f64> at fit+predict, 2·alpha denominator, the (1−x)·log(1−p) non-occurrence term folded into the GEMM (flp_delta = log p − log(1−p) + per-class neg_prob_sum bias)"
  - "ComplementNB<F> (NB-04): complement weights (feature_all_[j]+alpha−feature_count_[c,j]), optional second L1 norm, argmin decode (D-08) — labels match sklearn (not sign-flipped)"
  - "Shared discrete-NB free fns (D-03): decode_classes (integer/i32-range label guard) + resolve_class_log_prior (empirical / supplied / uniform), pub(crate) in multinomial_nb.rs, reused by the Bernoulli/Complement fits"
affects: [11-04, 11-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Discrete-NB GEMM joint-LL: raw = X @ feature_log_prob_.T via gemm(transb=true) over the (n_classes, n_features) stored buffer, host-add the per-class bias, then nb_common::log_sum_exp_normalize + argmax/argmin_decode (no new kernel)"
    - "BernoulliNB Pitfall-5 fold: store flp_delta = log p − log(1−p) as the GEMM operand and the per-class Σ_j log(1−p_cj) as an additive bias so the non-occurrence term needs no second matvec"
    - "ComplementNB argmin via negation: store sklearn's exact feature_log_prob_ (−logged / logged-normalized), compute jll = X@flp.T, decode labels with argmin_decode over −jll (== argmax over flp), proba log-sum-exp on jll directly"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs
    - crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs
    - crates/mlrs-algos/src/naive_bayes/complement_nb.rs
    - crates/mlrs-algos/tests/multinomial_nb_test.rs
    - crates/mlrs-algos/tests/bernoulli_nb_test.rs
    - crates/mlrs-algos/tests/complement_nb_test.rs

key-decisions:
  - "decode_classes + resolve_class_log_prior added pub(crate) in multinomial_nb.rs (next to the existing validate_discrete_alpha) and reused by the Bernoulli/Complement fits — function-level sharing (D-03), no shared base struct."
  - "ComplementNB stores sklearn's EXACT feature_log_prob_ (−logged default, logged/Σlogged under norm) so predict_proba matches sklearn's _joint_log_likelihood byte-for-byte; labels use argmin_decode over the negated jll (D-08 grep satisfied) which is identically argmax over feature_log_prob_."
  - "BernoulliNB carries a new neg_prob_sum_ field (per-class Σ_j log(1−p_cj)) so the non-occurrence constant is precomputed at fit and added as a predict bias (Pitfall 5) rather than recomputed per query row."
  - "force_alpha is #[allow(dead_code)] on all three structs — the D-06 clip already applied at build(); the field is retained as fitted-config provenance (matches the Wave-0 stub field set)."

patterns-established:
  - "The three count variants are independent structs sharing ONLY the GEMM joint-LL shape + the nb_common helpers + the two discrete free fns — D-03 holds (no NbBase). ComplementNB's weights/decision are verbatim from sklearn, NOT copied from MultinomialNB (Pitfall 6)."

requirements-completed: [NB-02, NB-03, NB-04]

# Metrics
duration: 20min
completed: 2026-06-21
---

# Phase 11 Plan 03: Count-based Naive Bayes (NB-02/03/04) Summary

**MultinomialNB / BernoulliNB / ComplementNB filled on the Wave-0 seam — three independent structs sharing the GEMM joint-LL path (`class_log_prior_ + X @ feature_log_prob_.T` via gemm transb=true) and the nb_common helpers, each with its own per-variant smoothing denominator and decision rule: alpha·n_features (MNB), 2·alpha + the folded non-occurrence term + binarize (BNB), complement weights + optional L1 norm + argmin (CNB). All three pass the exact-labels HARD gate on cpu f32+f64 with proba rows summing to 1, leak-free across re-fit.**

## Performance
- **Duration:** ~20 min
- **Tasks:** 3
- **Files modified:** 6 (3 source + 3 test)

## Accomplishments
- **MultinomialNB (NB-02):** `feature_log_prob_[c,j] = log((count+alpha)/(Σ_j count + alpha·n_features))` (Pitfall 4 — denominator smoothing is alpha·n_features), joint LL `class_log_prior_[c] + (X @ feature_log_prob_.T)[i,c]` via the device `gemm` (transb=true reads the stored `(n_classes, n_features)` buffer as its transpose), normalized by `log_sum_exp_normalize` + `argmax_decode`. 8/8 oracle tests green incl. `force_alpha_clip` (force_alpha=false & alpha=1e-12 builds, clipped to 1e-10).
- **BernoulliNB (NB-03):** `binarize: Option<f64>` applied to a host copy at fit AND predict (`Some(t)` → x>t, `None` → assume-binary pass-through); `feature_log_prob_[c,j] = log((count+alpha)/(class_count[c]+2·alpha))` (Pitfall 4 — 2·alpha); the `(1−x)·log(1−p)` non-occurrence term folded into the GEMM as `flp_delta = log p − log(1−p)` + a precomputed per-class `neg_prob_sum_ = Σ_j log(1−p_cj)` bias (Pitfall 5). 8/8 green incl. `binarize_none` (the None path on pre-binarized data reproduces the sklearn-default labels+proba).
- **ComplementNB (NB-04):** `feature_all_[j] = Σ_c feature_count_[c][j]`, `comp_count[c,j] = feature_all_[j] + alpha − feature_count_[c,j]`, `logged[c,j] = log(comp/Σ_j comp)`, stored `feature_log_prob_ = −logged` (default) or `logged / Σ_j logged` (norm) — verbatim sklearn (Pitfall 6, NOT copied from MNB). Labels via `argmin_decode` over `−jll` (D-08 — identically argmax over feature_log_prob_), proba log-sum-exp on jll. 8/8 green incl. `norm_true` (L1-normalized weight rows each sum to 1.0).

## Task Commits
1. **Task 1: MultinomialNB fit + GEMM joint-LL predict + oracle (NB-02)** — `5171820` (feat)
2. **Task 2: BernoulliNB fit (binarize + non-occurrence term) + oracle (NB-03)** — `f9bf26c` (feat)
3. **Task 3: ComplementNB fit (complement weights, norm, argmin) + oracle (NB-04)** — `a63c911` (feat)

_TDD note: each variant's source + filled oracle test landed in a single `feat` commit because the cpu gate (`cargo test`) is the RED→GREEN witness and the test+impl are file-disjoint per the Wave gate; the per-task acceptance grep + the green oracle are the gate evidence._

## Files Created/Modified
- `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs` — Filled `fit` (GATHER counts → flp with alpha·n_features denominator → class_log_prior); added the shared `decode_classes`/`resolve_class_log_prior` pub(crate) free fns (D-03); added the `joint_log_likelihood` GEMM evaluator + the three predict-trait impls + a `feature_log_prob` accessor.
- `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs` — Filled `fit` (binarize → GATHER → 2·alpha flp + neg_prob_sum); added the `neg_prob_sum_` field, `binarize_host`, the folded-GEMM `joint_log_likelihood`, the three predict traits, a `feature_log_prob_delta` accessor.
- `crates/mlrs-algos/src/naive_bayes/complement_nb.rs` — Filled `fit` (GATHER → feature_all_ → complement weights ± L1 norm); added the GEMM `joint_log_likelihood` (+ class_log_prior only in the single-class edge case), argmin-via-negation labels, the three predict traits, a `feature_log_prob` accessor.
- `crates/mlrs-algos/tests/{multinomial,bernoulli,complement}_nb_test.rs` — Un-ignored + filled all 8 cases each (exact_labels f32+f64, proba_band f32+f64, default_matches_sklearn, build_rejects_bad_alpha, the per-variant case, refit_releases_buffers).

## Decisions Made
- **Shared discrete free fns (D-03):** `decode_classes` (integer + i32-range label guard, WR-02) and `resolve_class_log_prior` (supplied / empirical / uniform) live `pub(crate)` in multinomial_nb.rs and are reused by the Bernoulli/Complement fits — function-level sharing, no base struct.
- **ComplementNB stores sklearn's exact `feature_log_prob_`** (−logged or logged-normalized) so `predict_proba` matches `_joint_log_likelihood` exactly; labels decode `argmin` over `−jll` (== argmax over flp), satisfying the D-08 `argmin_decode` grep AND the exact-label gate.
- **BernoulliNB `neg_prob_sum_` field** precomputes the per-class non-occurrence constant at fit (Pitfall 5) — the predict bias is a vector add, not a per-row recomputation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Critical functionality] Shared `decode_classes` / `resolve_class_log_prior` free fns**
- **Found during:** Task 1 (the plan's `<action>` lists "classes_ sort/dedup" and "class_log_prior_ (empirical when fit_prior=true & class_prior=None, else supplied/uniform)" but no helper existed — only `validate_discrete_alpha` was shipped in Wave 0).
- **Issue:** The integer/i32-range label decode and the three-way prior resolution are identical across all three discrete variants; inlining them three times would violate D-03's function-level-sharing intent and risk drift.
- **Fix:** Added both as `pub(crate)` free fns in multinomial_nb.rs (next to `validate_discrete_alpha`) and reused them from the Bernoulli/Complement fits.
- **Files modified:** multinomial_nb.rs (definitions), bernoulli_nb.rs + complement_nb.rs (call sites).
- **Commit:** `5171820` (defined), `f9bf26c` / `a63c911` (reused).

**2. [Rule 2 - Critical functionality] BernoulliNB `neg_prob_sum_` fitted field**
- **Found during:** Task 2 (the Wave-0 stub had no field for the per-class non-occurrence constant).
- **Issue:** Pitfall 5's folded GEMM needs the per-class `Σ_j log(1−p_cj)` constant at predict; without a stored field it would be recomputed from the weights every predict call.
- **Fix:** Added `neg_prob_sum_: Option<Vec<f64>>` to the struct, populated at fit, added as a predict bias.
- **Files modified:** bernoulli_nb.rs.
- **Commit:** `f9bf26c`.

---

**Total deviations:** 2 auto-fixed (both Rule 2 — the plan's `<action>` mandated these behaviors; the helpers/field are their natural implementation, not scope creep).

## Verification Evidence
- `cargo test --features cpu -p mlrs-algos --test multinomial_nb_test --test bernoulli_nb_test --test complement_nb_test` — **24 passed, 0 failed, 0 ignored** (8 each).
- Full NB suite (incl. nb_common + gaussian) — 9 + 7 + 24 = all green, 0 ignored, 0 failed.
- HARD gate: `predict_labels == sklearn predict` EXACTLY on f32 AND f64 for all three variants (integer equality, no band). ComplementNB labels match sklearn (the argmin convention is correct, NOT sign-flipped).
- `predict_proba` within band (f64 1e-5, f32 1e-3) AND every row sums to 1.0 ± 1e-6 for all three.
- Per-variant cases green: `force_alpha_clip` (MNB), `binarize_none` (BNB assume-binary path reproduces sklearn default), `norm_true` (CNB L1-normalized weight rows each sum to 1.0).
- Acceptance greps: `gemm` in multinomial_nb.rs (3), `binarize` in bernoulli_nb.rs (17), `argmin_decode` in complement_nb.rs (2).
- Non-comment `grep -c "SharedMemory\|F::INFINITY\|Atomic"` == 0 for all three (cpu-MLIR-safe; no new `#[cube]` kernel — only the v1 reduce GATHER + the gemm prim).
- `cargo build --features cpu -p mlrs-algos` — exits 0, zero warnings.

## Threat Flags
None — no new network/auth/file/schema surface. The plan's threat register (T-11-03-01..04) is mitigated: build() rejects `alpha<0` → InvalidAlpha + the force_alpha clip+warn (witnessed by `build_rejects_bad_alpha` / `force_alpha_clip`); counts accumulate in host f64 via class_grouped_sum (no integer overflow); alpha smoothing keeps the log arguments positive (host single-terminal-log, no NaN/Inf); the n_features-agreement geometry guard fires before any gemm launch (and the gemm prim u32-guards the grid).

## Known Stubs
None — all three fit bodies and predict surfaces are fully wired to the committed fixtures; no placeholder/empty-data paths remain.

## Self-Check: PASSED

All three source files + three test files verified on disk; all three task commits present in git history (5171820, f9bf26c, a63c911).

---
*Phase: 11-naive-bayes*
*Completed: 2026-06-21*
