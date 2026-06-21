---
phase: 11
slug: naive-bayes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-21
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `11-RESEARCH.md` §"Validation Architecture".

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `mlrs_core::{load_npz, OracleCase}` oracle harness; sklearn `.npz` fixtures |
| **Config file** | none — `cargo test` per crate; fixtures in `tests/fixtures/*.npz` (committed blobs) |
| **Quick run command** | `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test` (per-variant, targeted) |
| **Full suite command** | `cargo test --features cpu -p mlrs-algos` (NB) + `cargo test --features rocm -p mlrs-algos` (f32 gate) + `cargo test -p mlrs-py` (PyO3 smoke) |
| **Oracle regen (build-time only)** | `/tmp/oracle-venv/bin/python scripts/gen_oracle.py` (numpy/scipy/scikit-learn; blobs committed, CI never runs it — see memory `oracle-fixture-regen-needs-venv`) |
| **Estimated runtime** | ~seconds per targeted variant test; full `mlrs-algos` suite ~6 min (memory `backend-test-suite-slow` — background it) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --features cpu -p mlrs-algos --test <variant>_nb_test` (only the variant being edited — seconds, NOT the full mlrs-algos suite).
- **After every plan wave:** Run all five NB tests targeted (`--test gaussian_nb_test --test multinomial_nb_test --test bernoulli_nb_test --test complement_nb_test --test categorical_nb_test`) + `cargo test -p mlrs-py` smoke. Background the full backend suite if needed.
- **Before `/gsd-verify-work`:** All five NB oracle tests green on `--features cpu` (f64) AND `--features rocm` (f32, f64 skipped-with-log); PY-06 smoke green; exact-labels hard gate green for all five; every `predict_proba` row sums to 1; PoolStats no-leak per estimator.
- **Max feedback latency:** ~30 seconds (targeted per-variant test).

---

## Per-Task Verification Map

| Req ID | Behavior | Wave | Test Type | Automated Command | File Exists | Status |
|--------|----------|------|-----------|-------------------|-------------|--------|
| NB-01 | GaussianNB exact predict labels (HARD gate) | est | oracle unit | `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test exact_labels` | ❌ W0 | ⬜ pending |
| NB-01 | GaussianNB predict_proba band + rows-sum-to-1 | est | oracle unit | `… gaussian_nb_test proba_band` | ❌ W0 | ⬜ pending |
| NB-01 | `builder().build()` == sklearn default (`var_smoothing=1e-9`) | est | unit | `… gaussian_nb_test default_matches_sklearn` | ❌ W0 | ⬜ pending |
| NB-01 | build() rejects `var_smoothing < 0` (BuildError) | est | unit | `… gaussian_nb_test build_rejects_bad_var_smoothing` | ❌ W0 | ⬜ pending |
| NB-02 | MultinomialNB exact labels + proba band; densify path | est | oracle unit | `… multinomial_nb_test exact_labels` / `proba_band` | ❌ W0 | ⬜ pending |
| NB-02 | build() rejects `alpha < 0`; force_alpha clip+warn | est | unit | `… multinomial_nb_test build_rejects_bad_alpha` / `force_alpha_clip` | ❌ W0 | ⬜ pending |
| NB-03 | BernoulliNB exact labels (`(1−x)·log(1−p)`, binarize) | est | oracle unit | `… bernoulli_nb_test exact_labels` | ❌ W0 | ⬜ pending |
| NB-03 | BernoulliNB `binarize=None` (assume-binary) path | est | oracle unit | `… bernoulli_nb_test binarize_none` | ❌ W0 | ⬜ pending |
| NB-04 | ComplementNB exact labels (argmin, complement weights) | est | oracle unit | `… complement_nb_test exact_labels` | ❌ W0 | ⬜ pending |
| NB-04 | ComplementNB `norm=True` weight L1-normalize | est | oracle unit | `… complement_nb_test norm_true` | ❌ W0 | ⬜ pending |
| NB-05 | CategoricalNB exact labels (ragged `feature_log_prob_`) | est | oracle unit | `… categorical_nb_test exact_labels` | ❌ W0 | ⬜ pending |
| NB-05 | CategoricalNB `min_categories` padding (MinCategories enum) | est | oracle unit | `… categorical_nb_test min_categories` | ❌ W0 | ⬜ pending |
| NB-05 | fit() rejects negative / non-integer categorical input (AlgoError) | est | unit | `… categorical_nb_test fit_rejects_bad_input` | ❌ W0 | ⬜ pending |
| (all) | every f64 oracle case skips on rocm via `skip_f64_with_log` | (all) | gate | embedded in each `exact_labels` (f64) test | ❌ W0 | ⬜ pending |
| (all) | GATHER kernel path passes `--features cpu` launch | (all) | gate | any oracle test compiled+run with `--features cpu` (launch witness) | ❌ W0 | ⬜ pending |
| (all) | PoolStats memory gate per estimator (no leak across re-fit) | (all) | unit | `… <variant>_nb_test refit_releases_buffers` (PoolStats live_bytes assert) | ❌ W0 | ⬜ pending |
| PY-06 | each `#[pyclass]` fit/predict/predict_proba/predict_log_proba/score round-trips | PY | smoke | `cargo test -p mlrs-py --test pyclass_smoke_test` (extend) | ⚠️ extend | ⬜ pending |
| PY-06 | get_params/set_params sklearn-named knobs; f32/f64 dispatch; GIL release | PY | smoke | `cargo test -p mlrs-py` | ⚠️ extend | ⬜ pending |
| PY-06 | estimator_checks re-triaged across full v2 surface | PY | manual/integration | sklearn `check_estimator` (Python, end-of-phase) | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Coverage Strategy (5 variants × 2 dtypes)

- **Both dtypes per variant:** each variant gets `*_f32_seed42.npz` and `*_f64_seed42.npz` fixtures (gen_oracle.py `dtype` param, existing convention). f64 tests gated by `skip_f64_with_log`; f32 tests run a documented band.
- **Exact labels = hard gate (no band) for all five** — integer outputs, the primary correctness witness. The proba band is secondary (bounds last-bit drift).
- **f32-on-rocm band:** GaussianNB log-proba gets the widest documented band; the four discrete variants are integer-count-based and band tighter.
- **One small geometry per variant** (mirror SGD's `40×4`, `8` query rows) — well-separated classes so exact labels are unambiguous. The `_sgd_blobs` class-blob generator is reusable for the continuous (Gaussian) variant; the discrete variants need integer-count `X`, and the categorical variant needs integer-encoded features (new small generators).

---

## Wave 0 Requirements

- [ ] `crates/mlrs-algos/src/traits.rs` — add `PredictLogProba` trait (D-07)
- [ ] `crates/mlrs-algos/src/naive_bayes/nb_common.rs` — free functions: `log_sum_exp_normalize`, `empirical_class_log_prior`, `argmax_decode`, `argmin_decode`, `accuracy_score`, `class_grouped_sum` (the GATHER helper)
- [ ] `crates/mlrs-algos/src/error.rs` — NB `BuildError` variants (`InvalidVarSmoothing`, `InvalidClassPrior`, reuse `InvalidAlpha`) + `AlgoError` variants (`InvalidCategoricalInput`, prior-length mismatch)
- [ ] `scripts/gen_oracle.py` — `gen_{gaussian,multinomial,bernoulli,complement,categorical}_nb` + integer-count / categorical-encoded data generators; commit `tests/fixtures/*_nb_{f32,f64}_seed42.npz`
- [ ] `crates/mlrs-algos/tests/{gaussian,multinomial,bernoulli,complement,categorical}_nb_test.rs` — oracle harness (mbsgd_classifier_test.rs template)
- [ ] `crates/mlrs-py/src/estimators/naive_bayes.rs` + `crates/mlrs-py/src/lib.rs` registration (5 `add_class`) + extend `pyclass_smoke_test.rs`
- [ ] Framework install: none — `cargo test` is the framework; oracle regen needs `/tmp` venv only at fixture-gen time.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| sklearn `check_estimator` re-triage across full v2 surface | PY-06 | Requires a Python/sklearn environment with the built wheel; not part of the Rust oracle harness | Build the cpu wheel, `pip install`, run sklearn `check_estimator` against the five NB estimators; triage expected-skip vs real failures |
| `--features rocm` f32 band confirmation | NB-01…NB-05 | rocm runtime is the runnable GPU gate (gfx1100); f64 UNSUPPORTED on rocm (memory `rocm-is-runnable-gpu-gate`) | Run each variant's f32 oracle test under `--features rocm`; confirm documented band holds (GaussianNB widest) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
