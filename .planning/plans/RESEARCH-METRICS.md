# mlrs Metrics Surface — Planner-Ready Deep Dive (Roadmap Phase 24 "metrics" only)

**Agent:** Research (Spec-TDD workflow). **Date:** 2026-07-16.
**Scope (locked by coordinator):** the free-function metrics surface only — METR-01/02/03 of roadmap Phase 24. Classification: `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `log_loss`, `roc_auc_score`, `precision_recall_curve`. Regression: `r2_score`, `mean_squared_error`, `mean_absolute_error`. **Excludes** preprocessing / feature_extraction / model_selection (separate future phases).
**Companion:** `.planning/plans/RESEARCH.md` (PY-ENSEMBLE; kept intact). This file reuses that report's verified binding/oracle/validation facts.
**Evidence labels:** `[VERIFIED: CODEGRAPH …]` · `[VERIFIED: LOCAL …]` · `[VERIFIED: WEB …]` · `[INFERRED: …]` · `[UNVERIFIED: …]`

---

## 1. Executive Summary

The metrics surface is the **cleanest possible next feature** structurally: every listed metric is a small, host-side, O(n) reduction over already-materialized 1-D label/target vectors (or a 2-D probability matrix for `log_loss`). **None requires a device kernel, a BufferPool, `py.detach`, or the f64 guard** — they never upload to the device. This makes the binding dramatically simpler than any estimator wrapper (no `#[pyclass]`, no `Any<Name>` dtype enum, no typestate, no pool lock). `[INFERRED: metrics operate on host label/target vectors; the one precedent — `accuracy_score` — is already pure host Rust with no device touch, see §2]`

mlrs already computes **exactly one** of these internally: `mlrs_algos::naive_bayes::nb_common::accuracy_score(pred: &[i32], y_true: &[i32]) -> f64` — a host function used for the classifier `score`. `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/naive_bayes/nb_common.rs:160]` Everything else (`r2`, `mse`, `mae`, `confusion_matrix`, precision/recall/f1, `log_loss`, `roc_auc`, `precision_recall_curve`) is **not** computed anywhere in Rust; the estimator `score()` methods inherit sklearn's own `ClassifierMixin.score` (accuracy) / `RegressorMixin.score` (R²), which run host-side in Python on `self.predict(X)`. `[VERIFIED: LOCAL grep — no r2_score/mse/mae/confusion/roc in crates/**/*.rs; python shims subclass sklearn's ClassifierMixin/RegressorMixin, e.g. linear.py:15-20, neighbors.py:12,50,85]`

**Recommended shape:**
- **Rust:** a new host-only module `crates/mlrs-algos/src/metrics/{mod,classification,regression}.rs` — plain generic `fn`s over `&[i32]` labels / `&[F]` targets, no CubeCL, mirroring the `accuracy_score` precedent.
- **PyO3:** a new `crates/mlrs-py/src/metrics.rs` with `#[pyfunction]` free functions (mirror `johnson_lindenstrauss_min_dim`), taking plain `Vec<i32>`/`Vec<f64>` (PyO3 `extract`), returning scalars / `Vec<Vec<i64>>` (confusion) / tuples (PR-curve). Registered in `lib.rs` via `m.add_function(wrap_pyfunction!(...))`.
- **Shim:** a new `crates/mlrs-py/python/mlrs/metrics.py` of **free functions** (NOT classes, NOT `MlrsBase` subclasses) exposed as the `mlrs.metrics` submodule, sklearn-signature-faithful, returning numpy/scalars/tuples.
- **Sub-sequencing:** **Tier A (land first)** = the pure reductions: `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `r2_score`, `mean_squared_error`, `mean_absolute_error`, `log_loss`. **Tier B (land second)** = the ranking/threshold-sweep metrics `roc_auc_score` and `precision_recall_curve` (sort + trapezoid / threshold enumeration; more edge cases, tie handling, `multi_class` scope). See §5 + §9.

**Confidence:** HIGH that metrics are host-only and self-contained, that only `accuracy_score` pre-exists, and on the binding/oracle mechanics. MEDIUM on the exact sklearn semantics the Planner must pin (`average` defaults, `zero_division`, `mean_squared_error` `squared=` deprecation, `roc_auc` `multi_class`) — enumerated as open questions in §9.

---

## 2. Existing-Code Reuse (do NOT reimplement)

- **`nb_common::accuracy_score(pred: &[i32], y_true: &[i32]) -> f64`** — the fraction of exact matches `Σ[pred_i == y_true_i] / n`, with a length-mismatch panic guard. Host-only, no device. The new `metrics::classification::accuracy_score` should be this function (re-exported or moved) so there is one source. `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/naive_bayes/nb_common.rs:156-164]`
- Adjacent reusable host helpers in `nb_common.rs`: `argmax_decode` (:117), `argmin_decode` (:124), `log_sum_exp_normalize` (:72), `empirical_class_log_prior` (:99), `class_grouped_sum` (:199). `log_sum_exp_normalize` and the class-count logic are directly relevant to `log_loss` and `confusion_matrix` label bookkeeping. `[VERIFIED: CODEGRAPH nb_common.rs]`
- **`score()` is sklearn's, not mlrs's.** The Python shims subclass sklearn `ClassifierMixin`/`RegressorMixin` directly (`linear.py:15`, `neighbors.py:12`, `kernel_ridge.py:15`), so `.score()` = sklearn's host `accuracy_score`/`r2_score` on `self.predict(X)`. There is no mlrs R²/MSE/MAE anywhere to reuse — these are net-new. `[VERIFIED: LOCAL grep; python/mlrs/*.py mixin imports]`
- **Device reductions exist but are NOT needed here.** `mlrs-backend/src/prims/reduce.rs` (`sum`/`mean`/`min`/`max`/`l2_norm`/`row_reduce`/`column_reduce`/`argmin`/`argmax`/`argmax_rows`) and `mlrs-kernels/src/reduce.rs` (`reduce_sum_*`/`reduce_sumsq_*`/`argmax_shared`…) provide device sums, but metrics take tiny host vectors already read back from `predict`; uploading them to sum on-device would add a host↔device round-trip and sync for no benefit. Precedent for host-side math over already-read-back buffers: the covariance `pinvh` reassembly is host-side (`empirical_covariance.rs:414-427`) and `accuracy_score` is host-only. **Recommendation: host-only; do not route metrics through the reduce prims.** `[VERIFIED: LOCAL crates/mlrs-backend/src/prims/reduce.rs:89-360; crates/mlrs-kernels/src/reduce.rs:60-395; crates/mlrs-algos/src/covariance/empirical_covariance.rs:414-427]`

---

## 3. Module-Layout Recommendation (host vs device, which crate)

**Recommendation: host-only Rust, new module `crates/mlrs-algos/src/metrics/`.**

Rationale:
- Inputs are 1-D label/target vectors (length = n_samples) and, for `log_loss`, an `n × n_classes` probability matrix — all already host-materialized by the caller's `predict`/`predict_proba`. There is no matrix compute that benefits from a kernel.
- The one precedent (`accuracy_score`) lives in `mlrs-algos` as plain host Rust. Covariance/eig reassembly also do float math host-side in `mlrs-algos`. So host-only metrics violate no project rule; the "compute in CubeCL, generic over float+runtime" constraint (CLAUDE.md) governs **device algorithm kernels**, not scalar post-hoc metrics over predicted vectors. `[INFERRED: CLAUDE.md compute constraint targets device kernels; accuracy_score + pinvh host precedent]`
- Keeping metrics in `mlrs-algos` (not `mlrs-backend`/`mlrs-kernels`) means no backend feature, no CubeCL trait bounds for the label-only metrics.

Proposed files:
- `crates/mlrs-algos/src/metrics/mod.rs` — module index + shared enums (`Average { Binary, Macro, Micro, Weighted, None }`, a `ZeroDivision` policy) + shared label-bookkeeping (unique-class discovery, per-class TP/FP/FN counts).
- `crates/mlrs-algos/src/metrics/classification.rs` — `accuracy_score(&[i32],&[i32]) -> f64` (moved/re-exported from nb_common), `confusion_matrix(&[i32],&[i32], labels) -> Vec<Vec<i64>>`, `precision_score`/`recall_score`/`f1_score` (over TP/FP/FN with `Average` + `zero_division`), `log_loss(&[i32], &[F] proba, n_classes, eps) -> f64` (generic over `F` for the proba matrix). Tier B: `roc_auc_score`, `precision_recall_curve`.
- `crates/mlrs-algos/src/metrics/regression.rs` — `r2_score`, `mean_squared_error`, `mean_absolute_error`, generic `fn<F: Float>(&[F], &[F]) -> f64` (accumulate in f64 for the 1e-5 gate regardless of input dtype — the covariance-reassembly f64-accumulate precedent). `[VERIFIED: LOCAL empirical_covariance.rs:414-427 f64 accumulation then cast]`
- Register `pub mod metrics;` in `crates/mlrs-algos/src/lib.rs`. `[VERIFIED: LOCAL crates/mlrs-algos/src/lib.rs exists as crate root]`

Numeric-precision note: accumulate all sums in `f64` and clip `log_loss` probabilities to `[eps, 1-eps]` with `eps = 1e-15` (sklearn's default) so the ≤1e-5 gate holds for both f32 and f64 inputs. `[VERIFIED: WEB sklearn log_loss default clipping — https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html]`

---

## 4. PyO3 Free-Function Binding Pattern

**The project already exposes free functions via PyO3** — two of them, so the pattern is verified, not hypothetical:
- `johnson_lindenstrauss_min_dim(n_samples: f64, eps: f64) -> PyResult<usize>` with `#[pyfunction]`, mapping algos errors via `algo_err_to_py`. `[VERIFIED: LOCAL crates/mlrs-py/src/estimators/projection.rs:379-382]`
- `backend_supports_f64() -> bool` with `#[pyfunction]`. `[VERIFIED: LOCAL crates/mlrs-py/src/lib.rs:166-169]`
- Both registered with `m.add_function(wrap_pyfunction!(name, m)?)?;` in `_mlrs`. `[VERIFIED: LOCAL lib.rs:196,238]`

**Recommendation for metrics — plain `Vec` extraction, NOT the arrow capsule path.** The estimator ingress (`capsule_to_array` → `float_dtype` → `as_f32`/`as_f64` → `validated_f*` → device upload) exists for zero-copy X matrices bound for the device. Metrics inputs are small 1-D vectors that stay on the host, so accept them as PyO3-extracted `Vec<i32>` (labels) / `Vec<f64>` (targets, proba) directly:

```rust
// crates/mlrs-py/src/metrics.rs  (SKELETON — planner writes bodies)
#[pyfunction]
fn accuracy_score(y_true: Vec<i32>, y_pred: Vec<i32>) -> PyResult<f64> { … }

#[pyfunction]
fn confusion_matrix(y_true: Vec<i32>, y_pred: Vec<i32>, labels: Option<Vec<i32>>)
    -> PyResult<Vec<Vec<i64>>> { … }   // 2-D → list-of-lists → np.asarray shim-side

#[pyfunction]
fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> { … }

#[pyfunction]  // Tier A: precision/recall/f1 carry average + zero_division as strings/ints
fn precision_score(y_true: Vec<i32>, y_pred: Vec<i32>, average: &str, zero_division: f64)
    -> PyResult<f64> { … }

#[pyfunction]  // log_loss: flat proba + n_classes (row-major), or Vec<Vec<f64>>
fn log_loss(y_true: Vec<i32>, y_prob: Vec<f64>, n_classes: usize, eps: f64)
    -> PyResult<f64> { … }

#[pyfunction]  // Tier B: PR curve returns a 3-tuple of arrays
fn precision_recall_curve(y_true: Vec<i32>, probas_pred: Vec<f64>)
    -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> { … }
```

Why plain `Vec` over arrow capsules here:
- **No device, no pool, no GIL dance.** Metrics never upload; there is nothing to meter, no f64-incapable-backend concern, no `py.detach`/`lock_pool`. Dropping the arrow-capsule + device machinery removes ~all the estimator-binding ceremony. `[INFERRED: metrics are host-only]`
- **Integer-label ingress gap.** The arrow ingress only accepts `Float32Array`/`Float64Array` (`float_dtype` rejects everything else). Classification `y_true`/`y_pred` are integer labels; forcing them through the float capsule would mean casting labels to float. PyO3 `Vec<i32>` extraction sidesteps this cleanly. `[VERIFIED: LOCAL ingress.rs:112-118 float-only]`
- **Errors:** validation failures (length mismatch, empty input, single-class roc_auc) raise `PyValueError` directly (or reuse `algo_err_to_py` if the algos fns return `Result<_, AlgoError>` — recommended so the sklearn-parity error messages live once in `mlrs-algos`). `[VERIFIED: LOCAL crates/mlrs-py/src/errors.rs exists; projection.rs:381 algo_err_to_py precedent]`

**Output crossing:** scalar `f64`/`usize` cross natively; a 2-D `confusion_matrix` returns `Vec<Vec<i64>>` (PyO3 → list-of-lists, shim `np.asarray`s it); PR-curve returns a Rust tuple `(Vec<f64>, Vec<f64>, Vec<f64>)` (PyO3 → Python tuple of lists). `egress.rs` (`FloatResult`/`vec_f_to_py`) is **device-oriented** (takes a `DeviceArray` + pool) and is NOT the right tool for host metrics — return owned `Vec`s directly instead. `[VERIFIED: LOCAL egress.rs:32-68 device-only signatures]`

**Deviation to flag:** this introduces the first PyO3 surface that takes bulk data as plain `Vec` rather than an arrow capsule. It is justified (host-only, integer labels) but the Planner should record it as a conscious convention exception (see §9 Q7).

---

## 5. Python Shim Shape

**Recommendation: a new `crates/mlrs-py/python/mlrs/metrics.py` of free functions exposed as the `mlrs.metrics` submodule** — mirroring `sklearn.metrics`, NOT the estimator shim pattern.

- Free functions do NOT subclass `MlrsBase` and do NOT participate in `output_type` — they own no estimator, and sklearn's metrics return plain scalars / numpy arrays / tuples regardless of input container. So the `MlrsBase._normalize`/`_to_output`/`output_type` machinery does not apply. `[VERIFIED: LOCAL base.py:28-95 — MlrsBase surface is estimator-oriented (fit/predict); metrics have no fit]`
- Each shim function: `np.asarray(y_true).ravel()` with the right dtype (`np.int32` for labels, `np.float64` for targets/proba), light validation (shape agreement), call the `_mlrs` free function, wrap the return (scalar → Python float; confusion → `np.asarray(…, dtype=np.int64)`; PR-curve → tuple of `np.asarray`). Import `_mlrs` lazily via the package `_load_ext()` (same lazy pattern as `base.py::_ext`). `[VERIFIED: LOCAL base.py:99-117 _ext lazy import; __init__.py:108-128 _load_ext]`
- Signatures mirror sklearn exactly (parameter names + defaults), e.g. `accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)`, `precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', zero_division='warn')`, `mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')`, `r2_score(...)`, `log_loss(y_true, y_pred, *, eps='auto', normalize=True, sample_weight=None, labels=None)`, `roc_auc_score(...)`, `precision_recall_curve(y_true, probas_pred, *, pos_label=None)`. Unsupported kwargs (e.g. `sample_weight`) either raise `NotImplementedError` or are scoped out — decide per §9. `[VERIFIED: WEB sklearn.metrics signatures — https://scikit-learn.org/stable/api/sklearn.metrics.html]`
- **Namespace:** expose as `mlrs.metrics.<fn>` (submodule), matching `sklearn.metrics`. In `crates/mlrs-py/python/mlrs/__init__.py` add `from . import metrics` (submodule import), NOT individual names into `__all__` top-level — this keeps `mlrs.accuracy_score` from colliding with the estimator namespace and matches user muscle memory (`from mlrs.metrics import r2_score`). `[VERIFIED: LOCAL __init__.py:22-98 current top-level estimator exports; INFERRED submodule mirrors sklearn.metrics]`

---

## 6. Oracle + Degenerate Fixtures

**No metrics oracle generator exists today.** `scripts/gen_oracle.py` has generators for estimators/prims (`gen_covariance`, `gen_logistic`, `gen_argmin_tie`, etc.) but nothing for metrics. `[VERIFIED: LOCAL grep gen_oracle.py — no gen_metric/gen_accuracy/gen_r2]`

**Fixture mechanism (verified):** each `gen_*` builds arrays with numpy/scipy/sklearn and calls `np.savez(out_path, **named_arrays)` into `<repo>/tests/fixtures` (`_FIXTURE_DIR = <repo>/tests/fixtures`, `gen_oracle.py:41`), filename `case_dtype_seed42.npz` (e.g. `saxpy_f32_seed42.npz`); **scalars are stored as length-1 arrays** (e.g. `stat_acc_test[0]` in the RF fixtures). Rust tests load via `mlrs_core::load_npz` → `OracleCase::expect_f64(name)`; Python tests via `np.load(fixture_path(...))`. The `main()` block calls each generator and prints the written path. `[VERIFIED: LOCAL gen_oracle.py:41,135-152,3690-3760; crates/mlrs-algos/tests/random_forest_classifier_test.rs:87-91 load_npz/expect_f64]`

**How to add metric generators:** add `gen_metrics_classification(dtype, case)` / `gen_metrics_regression(dtype, case)` that compute sklearn references and `np.savez` a fixture per case. Recommended keying: one fixture per (family, case, dtype), e.g. `metrics_cls_binary_f64_seed42.npz`, `metrics_cls_multiclass_f64_seed42.npz`, `metrics_reg_f64_seed42.npz`, plus dedicated degenerate fixtures `metrics_cls_degenerate_*_f64_seed42.npz`. Named arrays: inputs (`y_true`, `y_pred`, `y_prob`/`y_score`) + one reference array per metric (`ref_accuracy`, `ref_confusion` (2-D), `ref_precision_macro`, `ref_recall_binary`, `ref_f1_micro`, `ref_log_loss`, `ref_roc_auc`, `ref_pr_precision`/`ref_pr_recall`/`ref_pr_thresholds`, `ref_r2`, `ref_mse`, `ref_mae`). Regen venv: `python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn` (PEP-668), same as the general estimator path. **Stamp the sklearn version** used, in the generator docstring (the repo does not stamp it elsewhere — an existing gap noted in the companion report). `[VERIFIED: LOCAL gen_oracle.py:15-17 venv instructions; UNVERIFIED exact sklearn version]`

**Mandatory degenerate cases (roadmap Phase 24 SC-1) — exact sklearn calls + kwargs + tolerance tier:** `[VERIFIED: WEB sklearn.metrics docs for each]`

| Degenerate case | sklearn reproduction (exact kwargs) | Expected reference | Tier |
|---|---|---|---|
| Empty class in `confusion_matrix` | `confusion_matrix(y_true, y_pred, labels=[0,1,2])` where class 2 never appears | full 3×3 with a zero row/col | **EXACT** (int counts) |
| Zero-division precision | `precision_score(y_true, y_pred, average='binary', zero_division=0)` with no predicted positives | `0.0` (per `zero_division`) | EXACT |
| Zero-division recall | `recall_score(..., zero_division=0)` with no true positives | `0.0` | EXACT |
| f1 zero-division | `f1_score(..., zero_division=0)` degenerate | `0.0` | EXACT |
| Single sample | `accuracy_score([1],[1])` / `[1],[0]` | `1.0` / `0.0` | EXACT |
| Constant target (r2 denominator 0) | `r2_score([5,5,5], y_pred)` | sklearn returns `0.0` (perfect) or negative; **pin sklearn's actual value** | ≤1e-5 (define exact behavior) |
| Perfect prediction | `r2_score(y,y)=1.0`, `mean_squared_error(y,y)=0.0`, `accuracy=1.0` | exact | EXACT / ≤1e-5 |
| Single-class `roc_auc_score` | `roc_auc_score([1,1,1], scores)` | sklearn **raises `ValueError`** ("Only one class present") | gate the **error**, not a value |
| Degenerate confusion (all one class) | `confusion_matrix([0,0],[0,0])` | `[[2]]` (1×1) | EXACT |
| `precision_recall_curve` trivial | `precision_recall_curve([0,1], [0.1,0.9])` | reference arrays | ≤1e-5 |
| `log_loss` clipping | `log_loss(y_true, y_prob)` with a `0.0`/`1.0` probability → clipped to `1e-15` | finite value | ≤1e-5 |

**Tolerance tier per metric:** `accuracy_score`, `confusion_matrix` → **EXACT** (integer / exact-fraction). `precision`/`recall`/`f1` → **EXACT** when the ratio is rational-in-integers, else ≤1e-5. `r2_score`/`mean_squared_error`/`mean_absolute_error`/`log_loss`/`roc_auc_score`/`precision_recall_curve` → **≤1e-5** (accumulate in f64). f32 fixtures may need `atol=1e-4` per the existing `_atol(fixture)` dtype-branch convention. `[VERIFIED: LOCAL crates/mlrs-py/python/tests/test_oracle_neighbors.py:23-24 _atol convention]`

---

## 7. Files to CREATE or MODIFY

**CREATE (Rust algos):**
- `crates/mlrs-algos/src/metrics/mod.rs` — module index, `Average` enum, `ZeroDivision` policy, shared label bookkeeping.
- `crates/mlrs-algos/src/metrics/classification.rs` — accuracy (move/re-export from nb_common), confusion, precision/recall/f1, log_loss; Tier B: roc_auc, precision_recall_curve.
- `crates/mlrs-algos/src/metrics/regression.rs` — r2, mse, mae.

**MODIFY (Rust algos):**
- `crates/mlrs-algos/src/lib.rs` — add `pub mod metrics;`.
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs` — optionally re-export `accuracy_score` from `metrics::classification` to keep one source (keep the `nb_common` path working for the NB `score`). `[VERIFIED: CODEGRAPH nb_common.rs:160 — the reuse seam]`

**CREATE (PyO3):**
- `crates/mlrs-py/src/metrics.rs` — the `#[pyfunction]` free functions (§4).

**MODIFY (PyO3):**
- `crates/mlrs-py/src/lib.rs` — `mod metrics;` + one `m.add_function(wrap_pyfunction!(metrics::<fn>, m)?)?;` per metric (mirror the `johnson_lindenstrauss_min_dim` / `backend_supports_f64` registration lines :196,:238). `[VERIFIED: LOCAL lib.rs:196,238]`

**CREATE (Python shim + tests):**
- `crates/mlrs-py/python/mlrs/metrics.py` — free-function shim (§5).
- `crates/mlrs-py/python/tests/test_oracle_metrics.py` — replays the new metric fixtures through the full binding path (mirror `test_oracle_neighbors.py`); degenerate cases incl. the single-class `roc_auc` `pytest.raises(ValueError)`.
- `crates/mlrs-py/tests/test_metrics.py` (Rust-side PyO3 integration, AGENTS.md §2) — smoke + error-path (length mismatch → ValueError), mirroring `crates/mlrs-py/tests/test_naive_bayes.py`. `[VERIFIED: LOCAL crates/mlrs-py/tests/ listing]`

**CREATE (Rust algos oracle tests):**
- `crates/mlrs-algos/tests/metrics_classification_test.rs`, `crates/mlrs-algos/tests/metrics_regression_test.rs` — load the fixtures, assert against sklearn refs at the §6 tiers, `skip_f64_with_log` gate on f64 cases (mirror `random_forest_classifier_test.rs`). `[VERIFIED: LOCAL random_forest_classifier_test.rs:202 skip_f64_with_log]`

**MODIFY (oracle generation + fixtures):**
- `scripts/gen_oracle.py` — add `gen_metrics_*` generators + `main()` calls; commit the new `tests/fixtures/metrics_*.npz`.

**MODIFY (Python namespace):**
- `crates/mlrs-py/python/mlrs/__init__.py` — `from . import metrics` (submodule, not top-level `__all__` entries).

**Enumerating tests that do NOT apply (important — free functions, not estimators):**
- `crates/mlrs-py/python/tests/test_params.py` (the AST **purity gate** + `get_params`/mutation tables) — applies only to estimators with `__init__`/`get_params`; **metrics free functions are exempt**. No change. `[VERIFIED: LOCAL test_params.py:12,53,255 — keyed on estimator classes/get_params]`
- `crates/mlrs-py/python/tests/test_shims.py` (mixin/attribute enumeration) — estimator-only; **exempt**. No change (unless a smoke that `mlrs.metrics` imports is desired). `[VERIFIED: LOCAL test_shims.py:74-188 estimator mixin checks]`
- `crates/mlrs-py/python/tests/test_estimator_checks.py` (sklearn `check_estimator` sweep) — estimator-only; **exempt**. `[VERIFIED: LOCAL — file runs check_estimator over estimator instances]`

---

## 8. Validation Commands + Dependency Versions

**Validation (verified against repo in the companion report):**
- Rust algos metric oracle tests: `cargo test -p mlrs-algos --features cpu` (f64 gate) and `cargo test -p mlrs-algos --features wgpu` (f32 gate). Metrics are host-only, so they compile/run under any backend feature; `--features cpu` is the primary f64 gate. `[VERIFIED: LOCAL crates/mlrs-algos/tests/*_perf_test.rs:8 cargo-test-with-feature headers; random_forest_classifier_test.rs]`
- PyO3 Rust integration: `cargo test -p mlrs-py --features cpu` (links libpython via `pyo3/auto-initialize` dev-dep; do NOT pass `extension-module`). `[VERIFIED: LOCAL crates/mlrs-py/Cargo.toml dev-dependencies + comments]`
- Python shim + oracle: `maturin develop -m crates/mlrs-py/pyproject/cpu.pyproject.toml` then `pytest crates/mlrs-py/python/tests/`. `[VERIFIED: LOCAL __init__.py:104; crates/mlrs-py/pyproject/cpu.pyproject.toml]`
- Fixture regen (only if adding fixtures): `python3 -m venv /tmp/oracle-venv && /tmp/oracle-venv/bin/pip install numpy scipy scikit-learn && /tmp/oracle-venv/bin/python scripts/gen_oracle.py`. `[VERIFIED: LOCAL gen_oracle.py:15-17]`
- No `justfile`/`Makefile`/CI in repo. `[VERIFIED: LOCAL ls — none]`

**Dependency versions (verified):** `pyo3 0.28.3` (pinned; do not bump — arrow-59 pyarrow feature transitive pin), `arrow 59.0.0`, `cubecl 0.10.0` (not needed for metrics), `bytemuck 1`, `thiserror 2`, `anyhow 1`, `npyz 0.9` (fixture reader). Rust toolchain `stable`. Python ≥3.12 (`abi3-py312`). Oracle venv: `numpy scipy scikit-learn` — **pin and record the sklearn version** in the generator docstring. `[VERIFIED: LOCAL Cargo.toml, Cargo.lock, rust-toolchain.toml]`

---

## 9. Impact / Risk + Open Questions

**Impact scope: additive, external-public (new `mlrs.metrics` submodule); NO estimator, kernel, backend, or algos-estimator changes.** Touches a new algos module + new PyO3 module + new shim module + `gen_oracle.py` + new fixtures + new tests. Lowest-coupling feature in the roadmap surface. `[INFERRED: host-only, free-function, no estimator/device edits]`

- **Must change:** `metrics/{mod,classification,regression}.rs`, `mlrs-algos/src/lib.rs`, `mlrs-py/src/metrics.rs`, `mlrs-py/src/lib.rs`, `python/mlrs/metrics.py`, `python/mlrs/__init__.py`, `scripts/gen_oracle.py`, new fixtures, new Rust + Python metric tests.
- **May change:** `nb_common.rs` (re-export `accuracy_score` for single-sourcing).
- **Verification only:** existing estimator tests (unchanged; metrics don't touch them).
- **Out of scope:** preprocessing, feature_extraction, model_selection, `sample_weight` (unless the Planner opts in), multilabel/multioutput beyond `multioutput='uniform_average'`.

**Risks (trigger → consequence → prevention → verify):**
1. **`mean_squared_error` `squared=` deprecation** — sklearn ≥1.4 removed `squared=False`; RMSE is now the separate `root_mean_squared_error`. Trigger: implementing a `squared` param. Consequence: signature drift from the installed sklearn. Prevention: implement `mean_squared_error` returning MSE only; do NOT add `squared`; note RMSE is out of scope. Verify against the installed sklearn signature. `[VERIFIED: WEB https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html]`
2. **`average` defaults** — `precision/recall/f1` default `average='binary'` (needs `pos_label`); multiclass fixtures require `macro`/`micro`/`weighted`. Trigger: wrong default → mismatched reference. Prevention: implement all of `binary|macro|micro|weighted|None`, default `binary`, and generate multiclass fixtures for each. Verify per-average fixture. `[VERIFIED: WEB sklearn precision_score docs]`
3. **`zero_division`** — precision/recall/f1 need the `zero_division` policy (`0`/`1`/`nan`); the degenerate fixtures depend on it. Prevention: carry `zero_division` explicitly; default to sklearn's `'warn'`→`0.0` numeric behavior. Verify degenerate fixtures.
4. **`roc_auc_score` / `precision_recall_curve` are ranking/threshold sweeps** (Tier B) — require a stable sort with tie handling + trapezoidal integration; more edge cases than the reductions. Prevention: **sub-sequence them after Tier A**; scope `roc_auc` to **binary** first (`multi_class` OvR/OvO deferred). Verify with tie-heavy fixtures. `[INFERRED: ranking metrics need sort+integrate; coordinator flagged]`
5. **Constant-target `r2_score`** — denominator zero; sklearn returns a defined value (0.0 or negative depending on version). Prevention: pin sklearn's actual output in the fixture, don't hand-derive. Verify degenerate fixture.
6. **Single-class `roc_auc`** — sklearn raises `ValueError`; the gate must assert the **error**, not a value. Prevention: `pytest.raises(ValueError)` + a Rust `Err` gate. Verify.
7. **Plain-`Vec` ingress convention exception** — first bulk-data PyO3 surface not using an arrow capsule (justified: host-only + integer labels). Prevention: document the exception; if the project mandates arrow ingress universally, fall back to float-capsule labels with a cast. Owner: Planner.
8. **f32 accumulation** — summing many terms in f32 can breach 1e-5. Prevention: accumulate in f64 always; f32 fixtures use `atol=1e-4`. Verify both dtypes.

**Open Questions (resolve before/at planning):**
- **Q1** `roc_auc_score` multiclass: support `multi_class='ovr'/'ovo'` in v1, or **binary-only** first? (Recommend binary-only, defer multiclass.) Owner: user.
- **Q2** `mean_squared_error`: confirm MSE-only (no `squared=`), RMSE out of scope, against the installed sklearn version. Owner: Planner.
- **Q3** `average` set to support for precision/recall/f1 (`binary|macro|micro|weighted|None`) + `pos_label` semantics. Owner: user.
- **Q4** `sample_weight`: support or `NotImplementedError`? (Recommend defer — not in METR success criterion.) Owner: user.
- **Q5** `log_loss`: `eps` handling (`'auto'` vs fixed `1e-15`) + optional `labels` param. Owner: Planner.
- **Q6** Constant-target `r2_score` and empty-input exact behavior — pin to which sklearn version? (Ties to the unstamped-version gap.) Owner: Planner.
- **Q7** PyO3 ingress: plain `Vec` (recommended) vs arrow-capsule-with-float-cast for labels — confirm the convention exception. Owner: Planner.
- **Q8** Namespace: `mlrs.metrics.<fn>` submodule (recommended, mirrors sklearn) vs top-level `mlrs.<fn>`. Owner: user.
- **Q9** Report path: this file is `.planning/plans/RESEARCH-METRICS.md` per the coordinator brief. Owner: workflow.

---

## 10. Traceability (symbols + paths cited)

**CodeGraph / algos:**
- `crates/mlrs-algos/src/naive_bayes/nb_common.rs` — `accuracy_score` (:160), `argmax_decode` (:117), `argmin_decode` (:124), `log_sum_exp_normalize` (:72), `empirical_class_log_prior` (:99), `class_grouped_sum` (:199).
- `crates/mlrs-algos/src/covariance/empirical_covariance.rs:414-427` — host f64-accumulate-then-cast precedent.
- `crates/mlrs-algos/src/typestate.rs:265-345` — `PredictProba` / `ScoreSamples` (context for proba inputs to log_loss).
- `crates/mlrs-backend/src/prims/reduce.rs:89-360`, `crates/mlrs-kernels/src/reduce.rs:60-395` — device reductions (deliberately NOT used).

**PyO3 / shim:**
- `crates/mlrs-py/src/estimators/projection.rs:379-382` (`johnson_lindenstrauss_min_dim` `#[pyfunction]`), `crates/mlrs-py/src/lib.rs:166-169` (`backend_supports_f64`), `:196,:238` (`m.add_function(wrap_pyfunction!)`).
- `crates/mlrs-py/src/ingress.rs:50-118` (arrow capsule + float-only dtype), `crates/mlrs-py/src/egress.rs:32-68` (device-only egress — not used), `crates/mlrs-py/src/errors.rs` (`algo_err_to_py`).
- `crates/mlrs-py/python/mlrs/base.py:28-117` (MlrsBase / lazy `_ext`), `__init__.py:22-143` (namespace / `_load_ext`), `neighbors.py`/`linear.py`/`kernel_ridge.py` (sklearn mixin `score` provenance).
- `crates/mlrs-py/python/tests/test_oracle_neighbors.py:1-70` (oracle-replay + `_atol` template), `test_params.py`/`test_shims.py`/`test_estimator_checks.py` (estimator-only enumerating gates — exempt).

**Oracle / config:**
- `scripts/gen_oracle.py:41` (`_FIXTURE_DIR`), `:135-152` (savez pattern), `:3690-3760` (`main()` dispatch), `:15-17` (regen venv).
- `crates/mlrs-algos/tests/random_forest_classifier_test.rs:87-91,202` (load_npz/expect_f64/skip_f64_with_log template).
- `Cargo.toml`, `Cargo.lock`, `crates/mlrs-py/Cargo.toml`, `rust-toolchain.toml`, `.planning/ROADMAP.md:216-231` (Phase 24 metrics SC).

**Web (sklearn semantics, accessed 2026-07-16):**
- `https://scikit-learn.org/stable/api/sklearn.metrics.html` (signatures), `.../mean_squared_error.html` (squared= deprecation), `.../log_loss.html` (eps clipping), `.../precision_score.html` (average/zero_division), `.../roc_auc_score.html` (multi_class, single-class ValueError).

**Tools used:** CodeGraph `codegraph_explore` + local Read/Grep/Bash for repo ground truth; WebSearch/WebFetch of official sklearn docs for metric semantics/defaults; Context7 not required (sklearn is the oracle spec, and its stable API docs are the authoritative source). PageIndex not invoked — repo `.planning/*` + source were directly readable.

---

## 11. Confidence Assessment

- **HIGH:** Only `accuracy_score` pre-exists (host, nb_common:160); all other metrics are net-new. Metrics are host-only and need no kernel/pool/guard. The `#[pyfunction]` free-function binding pattern (two verified precedents). The oracle `np.savez`/`load_npz`/`expect_f64` mechanism and fixture dir. Which enumerating tests are exempt. Dependency versions. Module-layout recommendation.
- **MEDIUM:** Exact sklearn semantics the Planner must pin — `average` defaults, `zero_division`, `mean_squared_error` `squared=` deprecation state, `log_loss` eps, constant-target `r2`. The plain-`Vec` vs arrow-capsule ingress decision. Tier-B (roc_auc / PR-curve) tie-handling detail.
- **LOW / UNVERIFIED:** The exact sklearn version to pin fixtures against (unstamped in-repo). Whether `roc_auc` multiclass is in v1 scope (user decision). Final namespace decision (`mlrs.metrics` submodule vs top-level).
