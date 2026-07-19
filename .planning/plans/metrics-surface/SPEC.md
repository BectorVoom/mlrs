---
title: mlrs sklearn Metrics Surface (classification + regression free functions)
status: draft
format: markdown
spec_version: 1
spec_revision: 3
updated_at: 2026-07-17T00:00:00Z
source_requirements:
  - "User request: implement features in cuML/sklearn not yet in mlrs (coverage-gap fill)"
  - "Roadmap Phase 24 metrics success criterion (METR-01, METR-02, METR-03) — .planning/ROADMAP.md:216-231"
locked_decisions:
  - "Target feature: sklearn metrics surface (user chose over PY-ENSEMBLE recommendation)"
  - "Scope: Tier A + Tier B, INCLUDING multiclass roc_auc_score (OvR + OvO)"
  - "sample_weight: SUPPORTED across every metric"
  - "Namespace: mlrs.metrics submodule (mirror sklearn.metrics)"
  - "Layout: host-only Rust in crates/mlrs-algos/src/metrics/; PyO3 free functions; no device kernel"
pageindex_update: "NOT APPLICABLE — PageIndex library holds external CubeCL/reference docs only; no mlrs project spec document exists to update. This SPEC.md is the authoritative local draft."
---

# mlrs sklearn Metrics Surface — Draft Specification

> Draft. Nothing here is approved/implemented. Feeds the Planner Agent (PLAN.md) and Plan Checker gate.
> Evidence labels: `[VERIFIED: CODEGRAPH …]` `[VERIFIED: LOCAL …]` `[VERIFIED: WEB …]` `[INFERRED: …]` `[UNVERIFIED: …]`.
> Full evidence in companion `../RESEARCH-METRICS.md`; ground-truth inventory in `../RESEARCH.md`.

## 1. Context

mlrs is a Rust rewrite of cuML with sklearn-compatible estimators that must match scikit-learn within 1e-5. The estimator surface is broad, but **there is no free-function metrics module**: mlrs computes exactly one metric internally — `mlrs_algos::naive_bayes::nb_common::accuracy_score(&[i32],&[i32]) -> f64` `[VERIFIED: CODEGRAPH crates/mlrs-algos/src/naive_bayes/nb_common.rs:160]` — and the Python estimator `.score()` methods inherit sklearn's own `ClassifierMixin`/`RegressorMixin` host math rather than mlrs code `[VERIFIED: LOCAL crates/mlrs-py/python/mlrs/linear.py:15, neighbors.py:12]`. Users cannot call `mlrs.metrics.r2_score(...)` etc.

This feature adds a host-only `mlrs.metrics` submodule mirroring `sklearn.metrics` for the classification + regression metrics in roadmap Phase 24's metrics criterion. Every metric is a small O(n) host reduction over already-materialized 1-D label/target vectors (or an n×n_classes probability matrix for `log_loss`); **none needs a device kernel, BufferPool, `py.detach`, or the f64 guard** `[INFERRED: RESEARCH-METRICS §1-3; accuracy_score is host-only precedent]`.

## 2. Scope and Non-Goals

### In scope (11 metrics + supporting infrastructure)

Classification: `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `log_loss`, `roc_auc_score` (binary **and** multiclass OvR/OvO), `precision_recall_curve`.
Regression: `r2_score`, `mean_squared_error`, `mean_absolute_error`.

Cross-cutting, in scope:
- `sample_weight` support on every metric (weighted variants). **Documented exception:** if the pinned sklearn version's `roc_auc_score(multi_class='ovo', sample_weight=...)` raises (to be probed at fixture-generation time — Issue 2 / Q10), the OvO path carves out `sample_weight` explicitly: the Rust `roc_auc_score_multiclass` OvO branch rejects `sw != None` with `MetricError`, matching sklearn. OvR keeps full `sample_weight` support. This is the one sanctioned carve-out from "sample_weight on every metric."
- `average ∈ {binary, macro, micro, weighted, None}` + `pos_label` for precision/recall/f1.
- `zero_division ∈ {0, 1, nan, "warn"}` for precision/recall/f1.
- `labels` parameter for `confusion_matrix`, `precision/recall/f1`, `log_loss` (reorder + subset) — each with its own reorder acceptance test.
- `multi_class ∈ {ovr, ovo}` + `average ∈ {macro, weighted}` for multiclass `roc_auc_score`.
- Regression metrics are **single-output only** (1-D `y_true`/`y_pred`).
- Mandatory degenerate fixtures (§6 acceptance).
- Rust host algos module, PyO3 free-function bindings, `mlrs.metrics` Python shim, oracle generators + committed fixtures, Rust + Python oracle tests.

### Non-goals (explicitly out)

- preprocessing / feature_extraction / model_selection surfaces (separate future phases). `[VERIFIED: LOCAL .planning/ROADMAP.md:216]`
- `root_mean_squared_error` / `mean_squared_error(squared=False)` — sklearn ≥1.4 split RMSE into a separate function and removed `squared=`; MSE-only here. `[VERIFIED: WEB scikit-learn mean_squared_error docs]`
- **Multioutput regression (2-D `y`) is a NON-GOAL** (downgraded from an earlier draft per Plan-Check Issue 3; RESEARCH-METRICS §9 recommended deferring it). `r2_score`/`mean_squared_error`/`mean_absolute_error` accept 1-D inputs only. The Python shim MUST raise `NotImplementedError` when given a 2-D `y_true`/`y_pred` or a non-default `multioutput` argument (fail-closed, never a silently-wrong `ravel()`ed value — `ravel`ing 2-D for r2 gives `1−ΣSSres/ΣSStot ≠ mean_k(1−SSres_k/SStot_k)`). No 2-D regression fixture. `multioutput='raw_values'` / `'variance_weighted'` also out.
- Multilabel-indicator inputs; `top_k_accuracy`, `balanced_accuracy`, `average_precision_score`, and any metric not listed above.
- Device/GPU acceleration of metrics (host-only by design).
- No estimator, kernel, backend, or existing-algos changes (except a re-export seam in `nb_common.rs`).

## 3. Dependencies

- **Rust cores already present (reuse, do not reimplement):** `nb_common::accuracy_score` (:160); adjacent host helpers `log_sum_exp_normalize` (:72), `class_grouped_sum` (:199), `argmax_decode` (:117). `[VERIFIED: CODEGRAPH nb_common.rs]`
- **f64-accumulate-then-cast precedent:** `covariance/empirical_covariance.rs:414-427` (accumulate in f64 for the 1e-5 gate). `[VERIFIED: LOCAL]`
- **PyO3 free-function precedent:** `johnson_lindenstrauss_min_dim` `#[pyfunction]` (`crates/mlrs-py/src/estimators/projection.rs:379-382`) + `backend_supports_f64` (`lib.rs:166-169`), registered via `m.add_function(wrap_pyfunction!(...))` (`lib.rs:196,238`). `[VERIFIED: LOCAL]`
- **Oracle mechanism:** `scripts/gen_oracle.py` `np.savez` into `<repo>/tests/fixtures` (`_FIXTURE_DIR`, :41); Rust `mlrs_core::load_npz` → `OracleCase::expect_f64`; Python `np.load`. Scalars stored as length-1 arrays. `[VERIFIED: LOCAL gen_oracle.py:41; random_forest_classifier_test.rs:87-91]`
- **Capability gate:** Rust `capability::skip_f64_with_log()`; Python `@requires_f64` from `conftest.py`; f32 fixtures may use `_atol(fixture)` (`atol=1e-4`). `[VERIFIED: LOCAL test_oracle_neighbors.py:20-24]`
- **Versions:** `pyo3 0.28.3` (pinned — do NOT bump), `arrow 59.0.0`, Rust `stable`, Python ≥3.12 (`abi3-py312`). Oracle venv `numpy scipy scikit-learn` — **exact sklearn version MUST be stamped** in the generator docstring (currently unstamped — Q6). `[VERIFIED: LOCAL Cargo.lock, rust-toolchain.toml]` `[UNVERIFIED: sklearn version]`
- **External oracle:** scikit-learn `sklearn.metrics` (authoritative reference for every value + error). `[VERIFIED: WEB]`

## 4. Typed Contracts

Rust host layer (`crates/mlrs-algos/src/metrics/`), all sums accumulated in `f64`:

```rust
// metrics/mod.rs — shared
pub enum Average { Binary, Macro, Micro, Weighted, None_ }   // None_ => per-class vector
pub enum ZeroDivision { Zero, One, Nan }                       // "warn" maps to Zero at the boundary
pub enum MultiClass { Ovr, Ovo }
// sample_weight is Option<&[f64]> on every fn; None => unit weights.

// metrics/classification.rs
pub fn accuracy_score(y_true: &[i32], y_pred: &[i32], sw: Option<&[f64]>, normalize: bool) -> Result<f64, MetricError>;
pub fn confusion_matrix(y_true: &[i32], y_pred: &[i32], labels: Option<&[i32]>, sw: Option<&[f64]>) -> Result<Vec<Vec<f64>>, MetricError>; // counts; f64 to carry weights, integral when sw=None
pub fn precision_score(y_true: &[i32], y_pred: &[i32], labels: Option<&[i32]>, pos_label: i32, average: Average, sw: Option<&[f64]>, zero_division: ZeroDivision) -> Result<PrfOut, MetricError>; // scalar or per-class
pub fn recall_score(/* same shape */) -> Result<PrfOut, MetricError>;
pub fn f1_score(/* same shape */) -> Result<PrfOut, MetricError>;
pub fn log_loss(y_true: &[i32], y_prob: &[f64], n_classes: usize, labels: Option<&[i32]>, sw: Option<&[f64]>, eps: f64, normalize: bool) -> Result<f64, MetricError>; // y_prob row-major n×n_classes; BadShape if y_prob.len() != y_true.len()*n_classes
pub fn roc_auc_score_binary(y_true: &[i32], y_score: &[f64], pos_label: i32, sw: Option<&[f64]>) -> Result<f64, MetricError>; // Err on single class
pub fn roc_auc_score_multiclass(y_true: &[i32], y_score: &[f64], n_classes: usize, multi_class: MultiClass, average: Average, sw: Option<&[f64]>) -> Result<f64, MetricError>;
pub fn precision_recall_curve(y_true: &[i32], probas_pred: &[f64], pos_label: i32, sw: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), MetricError>; // (precision, recall, thresholds)

// metrics/regression.rs — generic over input float, accumulate f64
pub fn r2_score<F: Float>(y_true: &[F], y_pred: &[F], sw: Option<&[f64]>) -> Result<f64, MetricError>;
pub fn mean_squared_error<F: Float>(y_true: &[F], y_pred: &[F], sw: Option<&[f64]>) -> Result<f64, MetricError>;
pub fn mean_absolute_error<F: Float>(y_true: &[F], y_pred: &[F], sw: Option<&[f64]>) -> Result<f64, MetricError>;

pub enum MetricError { LengthMismatch, EmptyInput, SingleClassRocAuc, BadShape, InvalidWeight, WeightedOvoUnsupported } // -> NEW metric_err_to_py -> PyValueError (a sibling of algo_err_to_py, which only takes AlgoError)
pub enum PrfOut { Scalar(f64), PerClass(Vec<f64>) }
```

PyO3 layer (`crates/mlrs-py/src/metrics.rs`) — `#[pyfunction]` free functions taking PyO3-extracted `Vec<i32>` (labels) / `Vec<f64>` (targets, proba, scores, sample_weight), returning scalars / `Vec<Vec<f64>>` / `(Vec<f64>,Vec<f64>,Vec<f64>)`; errors via a NEW `metric_err_to_py(MetricError) -> PyValueError` (sibling of `algo_err_to_py`, which only accepts `AlgoError`). **Plain-`Vec` ingress, NOT the arrow capsule** (host-only + integer labels; capsule ingress is float-only). `[VERIFIED: LOCAL ingress.rs:112-118; RESEARCH-METRICS §4]`

Python shim (`crates/mlrs-py/python/mlrs/metrics.py`) — sklearn-signature-faithful free functions; `np.asarray(...).ravel()` with the right dtype, validate shapes, call `_mlrs.<fn>`, wrap return (scalar→float; confusion→`np.asarray(dtype=int64/float64)`; PR-curve→tuple of arrays). Not `MlrsBase`, no `output_type`. `[VERIFIED: LOCAL base.py:28-95; RESEARCH-METRICS §5]`

## 5. Failure-Isolated Behavioral Specifications

Each spec has one behavioral responsibility with one primary failure cause. Infra specs (INFRA/BIND/SHIM/ORACLE) are the shared substrate the per-metric specs build on; a per-metric acceptance failure isolates to that metric's own value logic once infra specs pass.

### METR-INFRA-01 — Host metrics module scaffolding + shared label/weight bookkeeping
- **status:** draft
- **rationale/source:** RESEARCH-METRICS §3; needed by every classification metric.
- **preconditions:** `crates/mlrs-algos/src/lib.rs` compiles.
- **input:** label vectors `&[i32]`, optional `sample_weight: &[f64]`, optional `labels: &[i32]`.
- **output:** `Average`/`ZeroDivision`/`MultiClass`/`MetricError`/`PrfOut` types; shared functions: unique-class discovery (sorted, or `labels`-ordered), per-class weighted TP/FP/FN accumulation, length/weight validation returning `MetricError`.
- **dependencies:** none beyond std; f64-accumulate convention.
- **behavior (G/W/T):** Given equal-length `y_true`/`y_pred` (+ optional `sample_weight` of same length), When bookkeeping runs, Then it yields the sorted unique class set (or the provided `labels`) and per-class weighted TP/FP/FN; Given a length mismatch or negative/NaN weight, Then it returns `MetricError::{LengthMismatch,InvalidWeight}` (no panic).
- **invariants:** with `sample_weight=None`, weighted counts equal integer counts; with `labels` given, class order/contents follow `labels` exactly (including classes absent from data → zero counts).
- **acceptance:** Rust unit tests over hand-built vectors (incl. empty class via explicit `labels`, weighted counts).
- **out of scope:** any specific metric value.
- **traceability:** `[VERIFIED: CODEGRAPH nb_common.rs:160,199]`
- **open Qs:** Q3 (average set — fixed to all 5), Q7 (Vec vs capsule — fixed to Vec).

### METR-CLS-01 — accuracy_score (single-source with nb_common)
- **input:** `y_true,y_pred: &[i32]`, `sample_weight: Option<&[f64]>`, `normalize: bool`.
- **output:** `f64` — weighted fraction (or weighted count if `normalize=false`) of exact matches.
- **behavior:** Given labels, Then result equals `sklearn.metrics.accuracy_score` (EXACT for unweighted rational; ≤1e-5 weighted). The existing `nb_common::accuracy_score` becomes a thin re-export of `metrics::classification::accuracy_score(..., None, true)` so there is ONE source and the NB `score` path is unchanged. `[VERIFIED: CODEGRAPH nb_common.rs:160]`
- **acceptance:** oracle fixture (binary+multiclass, unweighted+weighted) + single-sample degenerate + NB `score` regression (unchanged).
- **tier:** EXACT (unweighted), ≤1e-5 (weighted/normalize).

### METR-CLS-02 — confusion_matrix
- **input:** `y_true,y_pred: &[i32]`, `labels: Option<&[i32]>`, `sample_weight: Option<&[f64]>`.
- **output:** `Vec<Vec<f64>>` (C×C; integral when unweighted).
- **behavior:** Given labels (+ optional explicit `labels` incl. a class never appearing), Then the matrix equals `sklearn.metrics.confusion_matrix` including full zero rows/cols for absent classes; row/col order follows sorted unique labels or the given `labels`.
- **acceptance:** empty-class fixture (`labels=[0,1,2]`, class 2 absent → 3×3 with zero row/col), all-one-class (`[[n]]`), weighted.
- **tier:** EXACT (counts); ≤1e-5 (weighted).

### METR-CLS-03 — precision_score  /  METR-CLS-04 — recall_score  /  METR-CLS-05 — f1_score
- Three separate specs (independent failure modes; f1 depends on P and R but is a distinct output).
- **input:** `y_true,y_pred: &[i32]`, `labels`, `pos_label: i32`, `average ∈ {binary,macro,micro,weighted,None}`, `sample_weight`, `zero_division ∈ {0,1,nan}`.
- **output:** `PrfOut` (scalar, or per-class vector when `average=None`).
- **behavior:** For each `average`, result equals the corresponding `sklearn.metrics.{precision,recall,f1}_score`. Given no predicted positives (precision) / no true positives (recall) / degenerate (f1), the `zero_division` policy applies exactly (`0`/`1`/`nan`).
- **acceptance:** per-`average` fixtures (binary + multiclass), zero-division degenerate (each metric), weighted, `average=None` per-class vector.
- **tier:** EXACT for rational-in-integers, else ≤1e-5.
- **note:** f1 is computed from the same weighted TP/FP/FN (harmonic mean), NOT from mlrs precision×recall floats, to avoid double-rounding.

### METR-CLS-06 — log_loss
- **input:** `y_true: &[i32]`, `y_prob: &[f64]` (row-major n×n_classes), `n_classes`, `labels: Option<&[i32]>`, `sample_weight`, `eps` (default `1e-15`), `normalize`.
- **output:** `f64`.
- **behavior:** Probabilities clipped to `[eps, 1-eps]`; result equals `sklearn.metrics.log_loss` (weighted cross-entropy). Given a `0.0`/`1.0` probability, clipping yields a finite value.
- **acceptance:** binary + multiclass fixtures, clipping degenerate (prob 0/1), weighted, `labels` reorder.
- **tier:** ≤1e-5.
- **open Qs:** Q5 (`eps='auto'` mapping — fixed to explicit `1e-15`; shim accepts `eps='auto'`→`1e-15`).

### METR-CLS-07 — roc_auc_score (binary)
- **input:** `y_true: &[i32]` (2 classes), `y_score: &[f64]`, `pos_label`, `sample_weight`.
- **output:** `Result<f64, MetricError>`.
- **behavior:** Rank-based AUC (stable sort, average-rank tie handling) equals `sklearn.metrics.roc_auc_score`; Given a single class present, returns `Err(SingleClassRocAuc)` → `ValueError`.
- **acceptance:** binary fixture (incl. tie-heavy scores), single-class error, weighted.
- **tier:** ≤1e-5 (value); error-gate for single-class.

### METR-CLS-08 — roc_auc_score (multiclass OvR/OvO)
- **input:** `y_true: &[i32]` (>2 classes), `y_score: &[f64]` (n×n_classes), `n_classes`, `multi_class ∈ {ovr,ovo}`, `average ∈ {macro,weighted}`, `sample_weight`.
- **output:** `Result<f64, MetricError>`.
- **behavior:** For each (`multi_class`,`average`) combo, equals `sklearn.metrics.roc_auc_score(..., multi_class=..., average=...)`. `sample_weight` supported on the **OvR** path (weighted fixture required). **OvO + sample_weight:** if the pinned sklearn rejects it, the OvO branch returns `MetricError::WeightedOvoUnsupported` for `sw != None`, matching sklearn (documented carve-out — §2, Issue 2).
- **acceptance:** 3-class fixtures for {ovr,ovo}×{macro,weighted}; **weighted OvR fixture** (`ref_roc_auc_ovr_macro_sw` etc.); OvO-with-`sw` either a weighted fixture (if sklearn supports it) or a `MetricError`/`ValueError` gate (if not — decided at TASK-02 probe); probability-rows-need-not-sum-to-1 handling per sklearn.
- **tier:** ≤1e-5 (values); error-gate for the OvO-weighted carve-out.
- **open Qs:** Q1 — multiclass IS in scope (user-locked). Q10 — probe OvO+sample_weight support at TASK-02 (owner: Planner).

### METR-CLS-09 — precision_recall_curve
- **input:** `y_true: &[i32]`, `probas_pred: &[f64]`, `pos_label`, `sample_weight`.
- **output:** `(Vec<f64> precision, Vec<f64> recall, Vec<f64> thresholds)`.
- **behavior:** Threshold sweep over sorted distinct scores equals `sklearn.metrics.precision_recall_curve`: `precision`/`recall` length = `thresholds.len()+1`, trailing `(1.0, 0.0)` sentinel point, thresholds ascending.
- **acceptance:** trivial + tie-heavy fixture, weighted; array-length + sentinel invariants.
- **tier:** ≤1e-5 (elementwise, aligned arrays).

### METR-REG-01 — r2_score  /  METR-REG-02 — mean_squared_error  /  METR-REG-03 — mean_absolute_error
- Three separate specs. **input:** `y_true,y_pred: &[F]` (1-D, single-output only), `sample_weight`.
- **output:** `f64`, accumulated in f64.
- **behavior:** Equal to `sklearn.metrics.{r2_score,mean_squared_error,mean_absolute_error}` on 1-D inputs. Constant-target `r2_score` (denominator 0) returns **sklearn's actual pinned value** (do not hand-derive). Perfect prediction: `r2=1.0`, `mse=0.0`, `mae=0.0`. Multioutput (2-D `y`) is a non-goal (§2) — the shim raises `NotImplementedError` for 2-D input.
- **acceptance:** standard fixture (f32+f64), constant-target r2 degenerate, perfect-prediction, weighted. Shim-level: 2-D `y` → `NotImplementedError`.
- **tier:** ≤1e-5 (f64), `atol=1e-4` (f32).

### METR-BIND-01 — PyO3 free-function surface
- **rationale:** every metric must be callable from `_mlrs`.
- **input/output:** as §4 PyO3 contract.
- **behavior:** Each `#[pyfunction]` extracts plain `Vec`s, calls the algos fn, maps `MetricError`→`PyValueError`, returns native/list/tuple; registered in `lib.rs` via `m.add_function(wrap_pyfunction!(...))`. Length-mismatch → `ValueError`.
- **acceptance:** `cargo test -p mlrs-py --features cpu` smoke + error-path per metric (`crates/mlrs-py/tests/test_metrics.py`).
- **note:** first bulk-data PyO3 surface taking plain `Vec` not arrow capsule — conscious documented exception (Q7).

### METR-SHIM-01 — mlrs.metrics Python submodule
- **behavior:** `crates/mlrs-py/python/mlrs/metrics.py` exposes sklearn-signature-faithful free functions; `from . import metrics` in `__init__.py` (submodule, NOT top-level `__all__`). Each normalizes inputs (`np.asarray().ravel()`, dtype), calls `_mlrs.<fn>`, wraps return. `sample_weight` passed through. **Fail-closed:** a 2-D `y_true`/`y_pred` or non-default `multioutput` on the regression metrics raises `NotImplementedError` (multioutput is a non-goal, §2); other unsupported inputs raise `ValueError`. Labels are cast to an integer numpy dtype before the `_mlrs` call so PyO3 `Vec<i32>` extraction succeeds.
- **acceptance:** `from mlrs.metrics import r2_score` importable; return types (float / np.ndarray / tuple) match sklearn; enumerating estimator gates (`test_params`/`test_shims`/`test_estimator_checks`) are **exempt** (free functions). `[VERIFIED: LOCAL RESEARCH-METRICS §7]`

### METR-ORACLE-01 — oracle generators + committed fixtures
- **behavior:** `scripts/gen_oracle.py` gains `gen_metrics_classification` / `gen_metrics_regression` producing sklearn references via `np.savez` into `tests/fixtures/metrics_*.npz`, with the **exact sklearn version stamped** in the generator docstring; `main()` calls them. Fixtures committed. Named arrays per §6.
- **acceptance:** generators run in the oracle venv and write the fixtures; Rust + Python oracle tests load and pass.
- **open Qs:** Q6 (sklearn version to pin) — Planner resolves before committing fixtures.

## 6. Acceptance Scenarios

Every behavioral spec above maps to at least one Red acceptance test. Consolidated gate matrix:

| Spec | Rust oracle test | Python oracle test | Degenerate fixtures | Tier |
|---|---|---|---|---|
| METR-CLS-01 accuracy | ✅ | ✅ | single-sample; NB-score unchanged | EXACT/≤1e-5 |
| METR-CLS-02 confusion | ✅ | ✅ | empty-class(labels), all-one-class | EXACT/≤1e-5 |
| METR-CLS-03/04/05 P/R/F1 | ✅ | ✅ | zero_division per metric; average=None | EXACT/≤1e-5 |
| METR-CLS-06 log_loss | ✅ | ✅ | prob 0/1 clipping | ≤1e-5 |
| METR-CLS-07 roc_auc binary | ✅ | ✅ | single-class → ValueError; ties | ≤1e-5 + err |
| METR-CLS-08 roc_auc multiclass | ✅ | ✅ | ovr/ovo × macro/weighted | ≤1e-5 |
| METR-CLS-09 pr_curve | ✅ | ✅ | trivial + ties; sentinel/length | ≤1e-5 |
| METR-REG-01/02/03 r2/mse/mae | ✅ (f32+f64) | ✅ | constant-target r2; perfect pred | ≤1e-5 / atol1e-4 |
| METR-BIND-01 pyo3 | `cargo test -p mlrs-py` | — | length-mismatch → ValueError | behavioral |
| METR-SHIM-01 shim | — | import + return-type | — | behavioral |
| METR-ORACLE-01 fixtures | generators run | fixtures load | all degenerate committed | infra |

Mandatory degenerate cases (roadmap Phase 24 SC-1) with exact sklearn reproduction are enumerated in `../RESEARCH-METRICS.md §6` and MUST each have a committed fixture (or an error-gate for single-class roc_auc).

## 7. Impact Scope

**Classification: additive, cross-module within new modules + external-public** (new `mlrs.metrics` submodule). `[INFERRED]`
- **CREATE:** `crates/mlrs-algos/src/metrics/{mod,classification,regression}.rs`; `crates/mlrs-py/src/metrics.rs`; `crates/mlrs-py/python/mlrs/metrics.py`; `crates/mlrs-algos/tests/metrics_classification_test.rs`, `metrics_regression_test.rs`; `crates/mlrs-py/python/tests/test_oracle_metrics.py`; `crates/mlrs-py/tests/test_metrics.py`; `tests/fixtures/metrics_*.npz`.
- **MODIFY:** `crates/mlrs-algos/src/lib.rs` (`pub mod metrics;`); `crates/mlrs-algos/src/naive_bayes/nb_common.rs` (re-export accuracy_score); `crates/mlrs-py/src/lib.rs` (`mod metrics;` + registrations); `crates/mlrs-py/python/mlrs/__init__.py` (`from . import metrics`); `scripts/gen_oracle.py` (generators + main dispatch).
- **UNCHANGED / verification-only:** all existing estimator code + tests; estimator-enumerating Python gates are exempt.
- **Impact class:** `local` at the algos layer, `external/public` at the Python surface.

## 8. Compatibility and Migration

Purely additive — no breaking change. The only edit to existing code is making `nb_common::accuracy_score` delegate to the new module (behavior-preserving; the NB `score` path must stay green). No serialized format, no estimator signature, no kernel touched. `pyo3 0.28.3` unchanged.

## 9. Risks and Open Questions

Risks (full detail with prevention/verify in `../RESEARCH-METRICS.md §9`):
1. `mean_squared_error` `squared=` deprecation → MSE-only, no `squared` param.
2. `average` defaults (`binary` needs `pos_label`) → implement all 5, generate per-average fixtures.
3. `zero_division` policy drives degenerate fixtures → carry explicitly.
4. roc_auc / pr_curve sort+tie handling → Tier B, land after Tier A; multiclass adds ovr/ovo edge cases.
5. Constant-target r2 → pin sklearn's actual value in the fixture.
6. Single-class roc_auc → gate the `ValueError`, not a value.
7. Plain-`Vec` ingress convention exception → documented.
8. f32 accumulation → accumulate in f64, f32 fixtures `atol=1e-4`.
9. `sample_weight` doubles the fixture matrix (weighted + unweighted per metric) → plan for it (user-locked in).

Open questions (owner-tagged; resolve at/before planning):
- **Q1 roc_auc multiclass** — RESOLVED: in scope (OvR+OvO, macro+weighted). [user-locked]
- **Q3 average set** — RESOLVED: support all 5 (binary,macro,micro,weighted,None). [assumed default, sklearn parity]
- **Q4 sample_weight** — RESOLVED: supported on all metrics. [user-locked]
- **Q8 namespace** — RESOLVED: `mlrs.metrics` submodule. [user-locked]
- **Q2 mean_squared_error** — CONFIRM MSE-only against installed sklearn signature. Owner: Planner.
- **Q5 log_loss eps** — fixed `1e-15`; shim maps `eps='auto'`→`1e-15`. Owner: Planner (confirm sklearn's current default).
- **Q6 sklearn version** — pin & STAMP the exact version producing fixtures. Owner: Planner (blocks fixture commit). `[UNVERIFIED]`
- **Q7 Vec vs arrow-capsule ingress** — plain `Vec` (documented exception). Owner: Planner.
- **Q10 OvO + sample_weight** — probe the pinned sklearn's `roc_auc_score(multi_class='ovo', sample_weight=...)` at TASK-02; if it raises, apply the §2 carve-out (Rust OvO rejects `sw!=None`). Owner: Planner. [Plan-Check Issue 2]

**Revision note (Plan-Check pass 1 → SPEC v1 revised):** multioutput downgraded to non-goal (Issue 3); OvO+sample_weight carve-out documented (Issue 2); `MetricError`→`metric_err_to_py` corrected (Issue found in §4); regression metrics constrained to 1-D. Plan-level fixes (load_npz float-cast, mod.rs stub pre-creation, weighted pr_curve/roc_auc fixtures, labels-reorder tests, empty-NaN assertion) are delegated to PLAN.md revision.

**Revision note (post-implementation code review → SPEC v3 revised):** an independent code review of the landed implementation (commit `0788e17`) found that `accuracy_score`, `confusion_matrix`, `precision_score`/`recall_score`/`f1_score`, `log_loss`, `precision_recall_curve`, `r2_score`, `mean_squared_error`, and `mean_absolute_error` all indexed `sample_weight[i]` with NO length/validity check — a too-short weight vector panicked (surfacing as an unhandled PyO3 `PanicException` instead of `ValueError`), and a too-long one was silently truncated with no error at all. `precision_score`/`recall_score`/`f1_score` additionally discarded `class_bookkeeping`'s own graceful `Err` via `.expect(...)`. **Fix:** every function in §4's typed contract now returns `Result<_, MetricError>` and validates via the existing `validate_weight` helper (the pattern `class_bookkeeping`/`roc_auc_score_binary`/`roc_auc_score_multiclass` already used) — no panic, no silent truncation, on any metric. The PyO3 layer propagates via `metric_err_to_py` uniformly. A separate cleanup fix consolidated the regression module's weighted-mean helper (previously duplicated three ways despite a doc-comment claiming it was shared) into one generic `weighted_mean` reused by `r2_score`/`mean_squared_error`/`mean_absolute_error`. New regression tests lock in the fixed contract at both the Rust and PyO3/Python boundaries. Verified: 83 Rust tests + 68 Python tests green.

## 10. Traceability and Sources

- Companion research: `../RESEARCH-METRICS.md` (metrics deep-dive), `../RESEARCH.md` (ground-truth estimator inventory).
- Roadmap: `.planning/ROADMAP.md:216-231` (Phase 24 metrics SC METR-01/02/03).
- Reuse seam: `crates/mlrs-algos/src/naive_bayes/nb_common.rs:160` `[VERIFIED: CODEGRAPH]`.
- PyO3 free-fn precedent: `crates/mlrs-py/src/estimators/projection.rs:379-382`, `crates/mlrs-py/src/lib.rs:166-169,196,238` `[VERIFIED: LOCAL]`.
- Oracle mechanism: `scripts/gen_oracle.py:41`; `crates/mlrs-algos/tests/random_forest_classifier_test.rs:87-91,202` `[VERIFIED: LOCAL]`.
- sklearn semantics: scikit-learn `sklearn.metrics` stable API docs (accessed 2026-07-16) `[VERIFIED: WEB]`.
- PageIndex: no mlrs spec document exists (library holds external CubeCL/reference docs only); this SPEC.md is the authoritative local draft.
