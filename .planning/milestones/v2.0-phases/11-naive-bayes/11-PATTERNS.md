# Phase 11: Naive Bayes - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 16 (5 estimators + nb_common + traits/error extensions + 5 oracle tests + 1 PyO3 file + lib.rs registration + gen_oracle.py + pyclass_smoke_test extension)
**Analogs found:** 16 / 16 (every new/modified file has a verified in-repo analog)

All analogs below were read against the LIVE tree (not just RESEARCH.md citations). Line numbers are current as of this mapping.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs` | estimator (model) | CRUD (fit→fitted state→predict) | `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` | exact (builder + classifier) |
| `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs` | estimator (model) | CRUD + GEMM joint-LL | `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` | exact |
| `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs` | estimator (model) | CRUD + GEMM joint-LL | `mbsgd_classifier.rs` (GEMM) + `kernel_density.rs` (`Option<f64>` knob) | exact + partial |
| `crates/mlrs-algos/src/naive_bayes/complement_nb.rs` | estimator (model) | CRUD + GEMM joint-LL (argmin) | `mbsgd_classifier.rs` | exact |
| `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs` | estimator (model) | CRUD (ragged host tables) | `mbsgd_classifier.rs` (shape) + `kernel_density.rs` (`MinCategories` enum) | exact + partial |
| `crates/mlrs-algos/src/naive_bayes/nb_common.rs` | utility (free functions) | transform (host f64) | `kernel_density.rs` `kde_log_norm`/`lgamma` free-fns + host log-sum-exp loop | role-match |
| `crates/mlrs-algos/src/naive_bayes/mod.rs` | config (module index) | n/a | `crates/mlrs-algos/src/linear/mod.rs` | exact |
| `crates/mlrs-algos/src/traits.rs` (MODIFY) | trait surface | n/a | existing `PredictProba` trait in same file | exact (same file) |
| `crates/mlrs-algos/src/error.rs` (MODIFY) | error enum | n/a | existing `BuildError`/`AlgoError` variants in same file | exact (same file) |
| `crates/mlrs-algos/src/lib.rs` (MODIFY) | config (crate index) | n/a | existing `pub mod density;` line | exact (same file) |
| `crates/mlrs-algos/tests/gaussian_nb_test.rs` | test | oracle request-response | `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` | exact |
| `crates/mlrs-algos/tests/multinomial_nb_test.rs` | test | oracle | `mbsgd_classifier_test.rs` | exact |
| `crates/mlrs-algos/tests/bernoulli_nb_test.rs` | test | oracle | `mbsgd_classifier_test.rs` | exact |
| `crates/mlrs-algos/tests/complement_nb_test.rs` | test | oracle | `mbsgd_classifier_test.rs` | exact |
| `crates/mlrs-algos/tests/categorical_nb_test.rs` | test | oracle | `mbsgd_classifier_test.rs` | exact |
| `crates/mlrs-py/src/estimators/naive_bayes.rs` | binding (5 #[pyclass]) | request-response (FFI) | `crates/mlrs-py/src/estimators/linear.rs` `PyMBSGDClassifier` (lines 831-1144) | exact |
| `crates/mlrs-py/src/estimators/mod.rs` (MODIFY) | config | n/a | existing `pub mod linear;` lines | exact (same file) |
| `crates/mlrs-py/src/lib.rs` (MODIFY) | binding registration | n/a | existing `m.add_class::<PyMBSGDClassifier>()?;` (line 244) | exact (same file) |
| `scripts/gen_oracle.py` (MODIFY) | utility (fixture gen) | batch | `gen_mbsgd_classifier` (1776-1865) + `_sgd_blobs` (1753-1773) + `main()` dispatch (2007-2162) | exact |
| `crates/mlrs-py/tests/pyclass_smoke_test.rs` (MODIFY) | test (smoke) | n/a | existing `all_twelve_estimators_construct_unfit` (lines 32-55) | exact (same file) |

---

## Pattern Assignments

### `crates/mlrs-algos/src/naive_bayes/gaussian_nb.rs` (estimator, CRUD)

**Analog:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` (the closest classifier-with-builder analog: `builder()`/`Default`/`build()->Result<_,BuildError>`, `classes_` remap, device-resident fitted state, `Fit`+`PredictLabels`+`PredictProba`).

**Imports pattern** (`mbsgd_classifier.rs:16-28`):
```rust
use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::traits::{Fit, PredictLabels, PredictProba};
```
For NB add `use mlrs_backend::prims::reduce::{row_reduce, column_reduce, ScalarOp, ReducePath};` (the GATHER substrate — see `kernel_density.rs:58`) and `use crate::traits::PredictLogProba;` (the new D-07 trait), and `use crate::naive_bayes::nb_common::*;`.

**Struct shape — device-resident fitted state, `None` until fit** (`mbsgd_classifier.rs:35-46`):
```rust
pub struct MBSGDClassifier<F> {
    config: SgdConfig,
    classes_: Vec<i64>,              // DISTINCT sorted labels (Pitfall 4)
    n_features: usize,
    coef_: Option<DeviceArray<ActiveRuntime, F>>,   // device-resident (D-03)
    intercept_: Option<DeviceArray<ActiveRuntime, F>>,
}
```
For GaussianNB the fitted state is `theta_` (means, n_classes×n_features), `var_` (variances), `class_prior_`/`class_count_`, the global `epsilon_` scalar — each `Option<DeviceArray>` or host `Vec<f64>` per the "host f64 small tensors" responsibility map. Keep `classes_: Vec<i64>` + `n_features: usize` verbatim.

**Builder + `Default` seeding sklearn defaults (D-02)** (`mbsgd_classifier.rs:54-56, 97-132`):
```rust
pub fn builder() -> MBSGDClassifierBuilder { MBSGDClassifierBuilder::default() }

#[derive(Debug, Clone, Copy)]
pub struct MBSGDClassifierBuilder { /* knob fields */ }

impl Default for MBSGDClassifierBuilder {
    fn default() -> Self { Self { alpha: 1e-4, /* sklearn defaults */ } }
}
```
GaussianNB builder fields are `var_smoothing: f64` (default `1e-9`) and `priors: Option<Vec<f64>>` (default `None`) — D-09 sklearn-mirrored names, NO `alpha`. Setter methods take `mut self -> Self` (`mbsgd_classifier.rs:136-199`).

**`build()` data-INDEPENDENT validation (D-05)** (`mbsgd_classifier.rs:218-286`):
```rust
pub fn build<F>(self) -> Result<MBSGDClassifier<F>, BuildError>
where F: Float + CubeElement + Pod {
    if !(self.alpha >= 0.0) {
        return Err(BuildError::InvalidAlpha { estimator: "mbsgd_classifier", alpha: self.alpha });
    }
    // ... more data-independent predicates ...
    Ok(MBSGDClassifier { /* fitted = None */ })
}
```
GaussianNB build() checks `var_smoothing >= 0` (new `BuildError::InvalidVarSmoothing`) and `priors` entries finite+non-negative (new `BuildError::InvalidClassPrior`). The discrete variants add `alpha >= 0` (reuse `BuildError::InvalidAlpha`) + the D-06 `force_alpha` clip+warn (see Shared Patterns).

**`Fit` impl — geometry guard, `classes_` sort∘dedup, fitted state, then GATHER** (`mbsgd_classifier.rs:289-396`):
```rust
fn fit(&mut self, pool, x, y, shape) -> Result<&mut Self, AlgoError> {
    let (n_samples, n_features) = shape;
    if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
        return Err(AlgoError::Prim(PrimError::ShapeMismatch { operand: "x", rows: n_samples, cols: n_features, len: x.len() }));
    }
    let y = y.ok_or(AlgoError::NotFitted { estimator: "...", operation: "fit (requires y)" })?;
    // host: distinct-sorted classes_ (Pitfall 4)
    let mut classes_: Vec<i64> = raw_labels.clone();
    classes_.sort_unstable();
    classes_.dedup();
    // ... compute fitted state ...
    self.classes_ = classes_;
    self.n_features = n_features;
    Ok(self)
}
```
The `classes_.sort_unstable(); classes_.dedup();` idiom (line 342-344) is the exact `classes_` inference NB reuses for ALL five. NB has no 2-class restriction — drop the `classes_.len() != 2` check (`mbsgd_classifier.rs:345-353`); keep the i32-range guard (359-369) so predicted labels survive the `as i32` cast.

**Re-fit buffer release (WR-07)** — `mbsgd_classifier.rs:389` (`yp_dev.release_into(pool)`) and `kernel_density.rs:241-243`:
```rust
if let Some(old) = self.x_fit_.take() { old.release_into(pool); }
```
Apply to every per-class scratch buffer in the GATHER loop and to the prior fitted state on re-fit (the PoolStats no-leak gate).

**GEMM joint-LL matvec (for the discrete variants)** (`mbsgd_classifier.rs:506-523`):
```rust
let raw = gemm::<F>(pool, x, (n_query, n_features), coef, (n_features, 1), false, false, None)?;
let bias = host_to_f64(intercept.to_host(pool)[0]);
let raw_host = raw.to_host(pool);
raw.release_into(pool);
```
For MultinomialNB/BernoulliNB/ComplementNB generalize the `coef`/`(n_features, 1)` operand to `feature_log_prob_`/`(n_features, n_classes)` (the `.T` is RESEARCH A1 — confirm `transpose_b=true` or transpose host-side), and host-add the per-class `class_log_prior_` bias instead of a scalar intercept.

**`PredictLabels` — host argmax/argmin decode** (`mbsgd_classifier.rs:399-426`):
```rust
let mut labels: Vec<i32> = vec![0i32; n_query];
for (r, label) in labels.iter_mut().enumerate() {
    *label = if margins[r] >= 0.0 { self.classes_[1] as i32 } else { self.classes_[0] as i32 };
}
Ok(DeviceArray::from_host(pool, &labels))
```
NB replaces the binary-margin sign with `nb_common::argmax_decode(joint_ll, &classes_)` (ComplementNB uses `argmin_decode`, D-08).

**`PredictProba` — host materialization of an `n_query × n_classes` matrix** (`mbsgd_classifier.rs:438-460`):
```rust
let mut proba: Vec<F> = vec![F::from_int(0i64); n_query * 2];
// ... per-row fill ...
Ok(DeviceArray::from_host(pool, &proba))
```
NB fills `n_query × n_classes` via `nb_common::log_sum_exp_normalize`. `PredictLogProba` returns the same buffer pre-exp (`joint_ll - logsumexp`).

---

### `crates/mlrs-algos/src/naive_bayes/multinomial_nb.rs` / `complement_nb.rs` (estimator, CRUD + GEMM)

**Analog:** identical to GaussianNB above, plus the GEMM joint-LL (`mbsgd_classifier.rs:506-523`). These three count-based variants share the GEMM path but stay SEPARATE structs (D-03 — no shared base). MultinomialNB/ComplementNB add `alpha`/`force_alpha`/`fit_prior`/`class_prior` builder knobs (D-09 names). ComplementNB additionally carries `norm: bool` and decodes with `argmin` internally (D-08) — do NOT special-case it in the trait.

**force_alpha clip+warn** — see Shared Patterns. The denominators differ per variant (Pitfall 4): implement each `feature_log_prob_` formula verbatim from FEATURES.md; do NOT copy MultinomialNB into ComplementNB (Pitfall 6 — complement counts + optional L1 norm + argmin sign).

---

### `crates/mlrs-algos/src/naive_bayes/bernoulli_nb.rs` (estimator, CRUD + GEMM)

**Analog:** GaussianNB/Multinomial shape + the `Option<f64>` knob precedent from `kernel_density.rs`.

**`Option<T>` polymorphic knob (D-04 `binarize`)** — modeled on `KernelDensity`'s typed-knob handling. The `BandwidthSpec` enum (`kernel_density.rs:111-119`) is the precedent for a value-shaped knob; for `binarize: Option<f64>` the simpler `Option` suffices: `None` disables binarization (assume binary), `Some(t)` thresholds `x > t`. Builder default `Some(0.0)` (D-02). The `(1-x)·log(1-p)` non-occurrence term folds into the GEMM via `flp = log p - log(1-p)` + a per-class constant `Σ_j log(1-p_cj)` (Pitfall 5).

---

### `crates/mlrs-algos/src/naive_bayes/categorical_nb.rs` (estimator, CRUD, ragged host tables)

**Analog:** GaussianNB shape + the **enum precedent** in `kernel_density.rs`.

**Dedicated value-shaped enum (D-04 `MinCategories`)** — `kernel_density.rs:108-119` `BandwidthSpec` is the exact precedent:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthSpec {
    Numeric(f64),
    Scott,
    Silverman,
}
```
Model `MinCategories::{ Infer, Uniform(usize), PerFeature(Vec<usize>) }` on this (note `PerFeature` needs `Clone`, not `Copy`). `feature_log_prob_` is a ragged `Vec<Vec<f64>>` (one matrix per feature, variable category count — Pitfall 7), NOT a single tensor. Validate non-negative-integer input at `fit` (new `AlgoError::InvalidCategoricalInput` or reuse `InvalidLabels`), guard the predict-time lookup index against `n_categories_j`.

---

### `crates/mlrs-algos/src/naive_bayes/nb_common.rs` (utility, free functions — D-03 NO struct)

**Analog:** the free-function host-f64 helpers in `kernel_density.rs` (`kde_log_norm` `434-467`, `lgamma` `473-500`) and the terminal-log host loop (`score_samples` `342-360`).

**Host log-sum-exp single-terminal-log idiom** (`kernel_density.rs:350-360`):
```rust
let mut out_host: Vec<F> = vec![F::from_int(0i64); n_query];
for r in 0..n_query {
    let s = host_to_f64(row_sum_host[r]);
    let log_density = s.ln() + log_norm - log_n;   // single log, applied ONCE at the end
    out_host[r] = f64_to_host::<F>(log_density);
}
```
`nb_common::log_sum_exp_normalize(joint_ll: &[f64], n_classes: usize) -> (Vec<f64>, Vec<f64>)` follows Pattern 3: per row `m = max_c; lse = m + log(Σ exp(ll-m)); proba = exp(ll-lse)`. All host f64, single terminal log (Pitfall 3/9 — never `±∞`/`F::INFINITY` on device → cpu-MLIR-safe).

**Free functions, no shared struct (D-03):** `nb_common` exposes ONLY free `fn`s — `log_sum_exp_normalize`, `empirical_class_log_prior`, `argmax_decode`, `argmin_decode`, `accuracy_score`, `class_grouped_sum` (the GATHER helper). No `NbBase` struct, no trait-object. The five estimators CALL these; they share code at the function level only.

**GATHER helper (`class_grouped_sum`)** — composes `column_reduce`/`row_reduce` (`mlrs_backend::prims::reduce`, imported exactly as `kernel_density.rs:58`) over host-grouped per-class row buffers. One owner per (class, feature) — a GATHER, never a scatter-add (Pitfall 1/2, ROADMAP #1). `release_into(pool)` each per-class scratch buffer (WR-07).

---

### `crates/mlrs-algos/src/naive_bayes/mod.rs` (config, module index)

**Analog:** `crates/mlrs-algos/src/linear/mod.rs` (module-level doc + `pub mod <estimator>;` per file):
```rust
pub mod coordinate_descent;
pub mod elastic_net;
pub mod lasso;
pub mod linear_regression;
```
NB mod.rs declares `pub mod nb_common;` + `pub mod gaussian_nb;` … `pub mod categorical_nb;` with a module-level doc explaining D-03 (free functions, five independent structs). Re-export the five estimator structs + `MinCategories`.

---

### `crates/mlrs-algos/tests/{gaussian,multinomial,bernoulli,complement,categorical}_nb_test.rs` (test, oracle)

**Analog:** `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` (the full oracle harness template).

**Fixture path helper + dtype casts** (`mbsgd_classifier_test.rs:77-100`) — copy verbatim:
```rust
fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}
```
plus `host_to_f64`/`f64_to`/`assert_band` (86-117).

**Exact-labels HARD gate + f64 skip gate** (`mbsgd_classifier_test.rs:193-232`) — the primary correctness witness for all five:
```rust
#[test]
fn exact_labels() {                       // f64 case
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {  // rocm skips, cpu runs (D-07)
        println!("... f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).expect("...");
    let predict_ref: Vec<i32> = case.expect_f64("predict").iter().map(|&v| v.round() as i32).collect();
    let (labels, _proba) = fit_gaussian::<f64>(&case);
    assert_eq!(labels, predict_ref, "GaussianNB f64 exact predict labels (HARD gate)");
}
```
The paired `exact_labels_f32` (193-209) runs on all backends (no skip). Mirror `proba_*` band tests (388-415) and `default_matches_sklearn` (420-436) + `build_rejects_bad_*` (440-463). GaussianNB log-proba gets the WIDEST f32 band (A4); the four discrete variants band tighter.

**`build()`-rejection test** (`mbsgd_classifier_test.rs:440-463`):
```rust
let bad_alpha = MultinomialNB::<f64>::builder().alpha(-1.0).build::<f64>().err();
assert!(matches!(bad_alpha, Some(BuildError::InvalidAlpha { alpha, .. }) if alpha == -1.0));
```
GaussianNB tests `var_smoothing < 0` → `BuildError::InvalidVarSmoothing`. CategoricalNB adds a `fit_rejects_bad_input` test (negative/non-integer → `AlgoError`).

**PoolStats no-leak gate** — add a `refit_releases_buffers` test per variant (the ROADMAP recurring memory gate; assert `pool` live_bytes unchanged across a re-fit).

---

### `crates/mlrs-py/src/estimators/naive_bayes.rs` (binding, 5 #[pyclass])

**Analog:** `crates/mlrs-py/src/estimators/linear.rs` `PyMBSGDClassifier` (lines 831-1144) — the full builder-classifier PyO3 wrapper.

**`any_estimator!` Unfit-arm enum (sklearn-named knobs verbatim, D-09)** (`linear.rs:831-840`):
```rust
crate::any_estimator! {
    any:   AnyMBSGDClassifier,
    algo:  mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier,
    unfit: { loss: String, penalty: String, alpha: f64, /* ... */ seed: u64 },
}
```
NB: `AnyGaussianNB` unfit holds `{ var_smoothing: f64, priors: Option<Vec<f64>> }` (NO alpha — D-09); the four discrete unfit arms hold `{ alpha: f64, force_alpha: bool, fit_prior: bool, class_prior: Option<Vec<f64>> }` plus their per-variant knobs (`binarize: Option<f64>`, `norm: bool`, `min_categories: ...`). `Option<Vec<f64>>` is a valid `unfit:` field type (the macro is type-generic).

**`#[pyclass]` + `#[new]` with sklearn-default signature (D-02)** (`linear.rs:881-960`):
```rust
#[pyclass(name = "MBSGDClassifier")]
pub struct PyMBSGDClassifier { inner: AnyMBSGDClassifier }

#[pymethods]
impl PyMBSGDClassifier {
    #[new]
    #[pyo3(signature = (loss = "hinge".to_string(), /* sklearn defaults */ seed = 0))]
    fn new(/* args */) -> Self { Self { inner: AnyMBSGDClassifier::Unfit { /* ... */ } } }
```
`PyGaussianNB::new` signature is `(var_smoothing = 1e-9, priors = None)` (D-02 defaults, D-09 names). Also add `unfit_default()` + `is_unfit()` (the smoke-test seam, `linear.rs:886-913`).

**`fit` — dtype dispatch + builder + GIL release (PY-06)** (`linear.rs:966-1047`):
```rust
let fitted = py.detach(|| -> PyResult<AnyMBSGDClassifier> {
    let mut pool = crate::lock_pool();                       // poison-recovering lock (WR-04)
    match dt {
        FloatDtype::F32 => {
            let xd = validated_f32(as_f32(&xa)?, &mut pool)?;
            let mut est = MBSGDClassifier::<f32>::builder()
                .alpha(alpha) /* ... */
                .build::<f32>().map_err(build_err_to_py)?;   // D-09 BuildError → ValueError
            est.fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?;
            Ok(AnyMBSGDClassifier::F32(est))
        }
        FloatDtype::F64 => {
            crate::capability::guard_f64()?;                 // D-04 BEFORE upload
            /* ... f64 arm ... */
        }
    }
})?;
self.inner = fitted;
```
For NB the builder chain uses the sklearn-mirrored setters per estimator (D-09 — zero name translation). MultinomialNB's `fit` densifies sparse X at ingress (NB-02, PROJ-02 precedent — densify at the PyO3 boundary). Any `TryFrom<&str>` enum-string knob parses BEFORE `py.detach` with `map_err(build_err_to_py)` (`linear.rs:992-994`) — NB has none of these except a possible `min_categories` string path (most NB knobs are float/bool/Option).

**predict / predict_proba / dtype-suffixed accessors** (`linear.rs:1050-1095`):
```rust
fn predict_labels(&self, py, x, rows, cols) -> PyResult<Vec<i32>> {
    py.detach(|| {
        let mut pool = crate::lock_pool();
        match &self.inner {
            AnyMBSGDClassifier::F32(est) => Ok(est.predict_labels(&mut pool, &xd, (rows, cols)).map_err(algo_err_to_py)?.to_host_metered(&mut pool)),
            /* F64, Unfit→not_fitted */
        }
    })
}
```
Add `predict_log_proba_f32`/`_f64` (the new D-07 surface) alongside `predict_proba_f32`/`_f64` (dtype-suffixed because the return is `Vec<F>`). Add `score(x, y)` calling `nb_common::accuracy_score` after `predict_labels` (D-07). `classes_`/`is_fitted`/`dtype` accessors copy `linear.rs:1097-1143`.

---

### `crates/mlrs-py/src/lib.rs` (MODIFY — registration) + `estimators/mod.rs`

**Analog:** existing registration block (`lib.rs:202-247`):
```rust
use estimators::linear::{ /* ... */ };
m.add_class::<PyMBSGDClassifier>()?;
```
Add `use estimators::naive_bayes::{PyGaussianNB, PyMultinomialNB, PyBernoulliNB, PyComplementNB, PyCategoricalNB};` + five `m.add_class::<…>()?;` lines. Add `pub mod naive_bayes;` to `estimators/mod.rs` (alphabetical, after `linear`). PY-06 is the final cross-cutting sign-off: all v2 estimators registered, dtype accessors complete, `estimator_checks` re-triaged.

---

### `scripts/gen_oracle.py` (MODIFY — fixture generators)

**Analog:** `gen_mbsgd_classifier` (1776-1865) + `_sgd_blobs` (1753-1773) + the `main()` dispatch (2007-2162).

**Class-blob generator** (`gen_oracle.py:1753-1773`) — directly reusable for the CONTINUOUS GaussianNB; the discrete variants need integer-count X (new small generator) and CategoricalNB needs integer-encoded features (new generator):
```python
def _sgd_blobs(seed, n_classes=2):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, SGD_N_FEATURES)) * 4.0   # well-separated
    # ... build X, y, Xq ...
    return rng, x, y, xq
```

**Per-estimator generator shape** (`gen_oracle.py:1776-1865`) — fit sklearn, cast in fixture dtype, savez the named arrays:
```python
def gen_gaussian_nb(seed=SEED, dtype=np.float32):
    from sklearn.naive_bayes import GaussianNB
    _, x, y, xq = _sgd_blobs(seed, n_classes=2)
    def c(arr): return np.ascontiguousarray(np.asarray(arr)).astype(dtype)
    dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
    clf = GaussianNB().fit(x, y)
    out_path = os.path.join(_FIXTURE_DIR, f"gaussian_nb_{dtype_tag}_seed{seed}.npz")
    np.savez(out_path, X=c(x), Xq=c(xq), y=c(y),
             predict=c(clf.predict(xq)), predict_proba=c(clf.predict_proba(xq)))
    return out_path
```
Store `X`/`Xq`/`y`/`predict`/`predict_proba` (and per-variant fitted attrs as needed). `predict` is the exact-label HARD gate; `predict_proba` the band gate. **Register in `main()`** (the `for dtype in (np.float32, np.float64): print(f"wrote {gen_*(dtype=dtype)}")` block, 2150-2162) — both dtypes per variant. Regen needs the `/tmp` numpy/sklearn venv (project memory `oracle-fixture-regen-needs-venv`); blobs are committed, CI never runs this.

---

### `crates/mlrs-py/tests/pyclass_smoke_test.rs` (MODIFY — extend)

**Analog:** `all_twelve_estimators_construct_unfit` (lines 32-55):
```rust
#[test]
fn all_twelve_estimators_construct_unfit() {
    assert!(PyLinearRegression::unfit_default().is_unfit(), "LinearRegression");
    // ...
}
```
Add a `five_naive_bayes_estimators_construct_unfit` test asserting each `Py{Gaussian,Multinomial,Bernoulli,Complement,Categorical}NB::unfit_default().is_unfit()`. RESEARCH also flags a fit/predict/predict_proba/predict_log_proba/score round-trip smoke per estimator (PY-06).

---

## Shared Patterns

### Builder + split validation (D-01/D-05)
**Source:** `crates/mlrs-algos/src/linear/mbsgd_classifier.rs:54-286` (`builder()` → `Default` sklearn defaults → `build() -> Result<_, BuildError>`).
**Apply to:** all five NB estimator files.
```rust
pub fn builder() -> XBuilder { XBuilder::default() }
impl Default for XBuilder { fn default() -> Self { /* sklearn defaults (D-02) */ } }
pub fn build<F>(self) -> Result<X<F>, BuildError> {
    if !(self.alpha >= 0.0) { return Err(BuildError::InvalidAlpha { estimator, alpha: self.alpha }); }
    Ok(X { /* fitted = None */ })
}
```
Data-INDEPENDENT predicates at `build()` (D-05): `alpha >= 0`, `var_smoothing >= 0`, `min_categories` entries non-negative, `priors`/`class_prior` entries finite+non-negative. Data-DEPENDENT (`prior len == n_classes`, categorical input, `n_features` agreement) stay at `fit() -> AlgoError`.

### force_alpha clip + warn (D-06)
**Source:** RESEARCH §"Code Examples" (the data-independent clip lives at `build()`, modeled on the `mbsgd_classifier.rs:244-253` non-finite-reject precedent and the `log::warn!` host-side convention).
**Apply to:** MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB build().
```rust
let alpha = if !self.force_alpha && self.alpha < 1e-10 {
    log::warn!("alpha too small, setting alpha=1e-10. Use force_alpha=True to keep alpha unchanged.");
    1e-10
} else { self.alpha };
```
Parity depends only on the clipped numeric `1e-10`, not the warning text (A2).

### Error handling — two-tier (D-05), new variants
**Source:** `crates/mlrs-algos/src/error.rs` — `BuildError` (367-490, data-independent) and `AlgoError` (29-347, data-dependent), both `thiserror` with `estimator: &'static str` + value field (the project convention, NOT `anyhow` in libs).
**Apply to:** error.rs MODIFY + every NB file (raised via `?`).
Existing reusable: `BuildError::InvalidAlpha` (373-379), `AlgoError::NotFitted` (65-73), `AlgoError::InvalidLabels` (331-339), `AlgoError::Prim(#[from] PrimError)` (345-346 — geometry via `ShapeMismatch`/`DimMismatch`). New variants to add (each follows the exact `#[error("...")] Variant { estimator, value }` shape):
- `BuildError::InvalidVarSmoothing { estimator, var_smoothing: f64 }`
- `BuildError::InvalidClassPrior { estimator }` (or carry the offending value)
- `BuildError::InvalidMinCategories { estimator }`
- `AlgoError::InvalidCategoricalInput { estimator, reason: String }` (or reuse `InvalidLabels`)
- prior-length-mismatch: reuse `AlgoError::InvalidLabels` or add a dedicated variant.

### PyO3 boundary — GIL release + error mapping (PY-06/D-09)
**Source:** `crates/mlrs-py/src/estimators/linear.rs:995-1047` (`py.detach` + `crate::lock_pool`) and `crates/mlrs-py/src/errors.rs:56-89` (`algo_err_to_py` 56, `build_err_to_py` 71, `not_fitted` 84 — all map to `PyValueError`).
**Apply to:** all five #[pyclass] fit/predict/accessor bodies.
```rust
let out = py.detach(|| {
    let mut pool = crate::lock_pool();   // poison-recovering (WR-04) — NOT global_pool().lock().expect(..)
    /* float_dtype dispatch; guard_f64()? on F64 arm BEFORE upload; build().map_err(build_err_to_py)?; fit().map_err(algo_err_to_py)? */
});
```
`BuildError` (+ enum `TryFrom` failures) → `ValueError` at construction-time; `fit`-time `AlgoError` → `ValueError` via the existing single-site mappers. No new mapper needed (D-09).

### Host f64 single-terminal-log + reduce-prim GATHER (cpu-MLIR-safe)
**Source:** `crates/mlrs-algos/src/density/kernel_density.rs:58` (reduce import), `:331-360` (device row-reduce + host terminal log), `:434-500` (host f64 free functions).
**Apply to:** `nb_common.rs` and every NB `fit` (the GATHER) / `predict_*` (log-sum-exp).
NB writes NO new `#[cube]` kernel — only the validated `reduce` (host-segmented, Shared path, cpu-safe) and `gemm` prims. log-sum-exp/argmax/argmin/the class-conditional sums are host f64. Never `F::INFINITY`/SharedMemory/scatter-add (Pitfall 1/2, cubecl-cpu-no-shared-memory).

---

## No Analog Found

None. Every new/modified file has a verified in-repo analog. The two structurally-novel pieces — the GATHER `class_grouped_sum` helper and CategoricalNB's ragged `Vec<Vec<f64>>` `feature_log_prob_` — are COMPOSITIONS of existing analogs (the `reduce` prim host-segmentation + the `BandwidthSpec` enum precedent), not new patterns, so they are covered above rather than listed here.

---

## Metadata

**Analog search scope:**
- `crates/mlrs-algos/src/{linear,density}/` (estimator + enum/log-sum-exp precedents)
- `crates/mlrs-algos/src/{traits.rs,error.rs,lib.rs}` (trait/error/index surfaces to extend)
- `crates/mlrs-algos/tests/` (oracle harness template)
- `crates/mlrs-py/src/{lib.rs,dispatch.rs,errors.rs}` + `crates/mlrs-py/src/estimators/{linear.rs,mod.rs}` (PyO3 wrapper + macro + mappers)
- `crates/mlrs-py/tests/pyclass_smoke_test.rs` (smoke template)
- `scripts/gen_oracle.py` (fixture generators + main dispatch)

**Files scanned (read in full or targeted):** mbsgd_classifier.rs (573), traits.rs (270), error.rs (490), lib.rs (60), kernel_density.rs (500), dispatch.rs (115), estimators/linear.rs (PyMBSGDClassifier 816-1144), errors.rs (56-89), mbsgd_classifier_test.rs (464), gen_oracle.py (_sgd_blobs/gen_mbsgd_classifier/main), pyclass_smoke_test.rs (82), linear/mod.rs.

**Key cross-cutting verifications:**
- `any_estimator!` macro (dispatch.rs) is a SKELETON emitting only the `Unfit/F32/F64` enum; the `#[pymethods]` fit/predict bodies are HAND-WRITTEN per estimator (copy `PyMBSGDClassifier`, not the macro).
- `build_err_to_py`/`algo_err_to_py` both map to `PyValueError` (single-site, D-09) — no new mapper for NB.
- `classes_.sort_unstable(); classes_.dedup();` (mbsgd_classifier.rs:342-344) is the reusable `classes_` inference for ALL five NB variants (NB drops the 2-class restriction).

**Pattern extraction date:** 2026-06-21
