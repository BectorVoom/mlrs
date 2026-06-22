# Phase 11: Naive Bayes - Research

**Researched:** 2026-06-21
**Domain:** sklearn-compatible Naive Bayes classifiers (5 variants) in Rust over CubeCL reduce prim + PyO3
**Confidence:** HIGH (all stack/pattern claims verified against in-repo source; per-variant math pinned by FEATURES.md)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (inherited):** All five NB estimators constructed via the **builder pattern** — `Estimator::builder().setter(..).build()?` — NOT `new(positional)`/`with_opts()`.
- **D-02 (inherited):** `Estimator::builder().build()` with no setters MUST reproduce scikit-learn's **default** estimator. Per-variant defaults: GaussianNB `var_smoothing=1e-9`, `priors=None`; MultinomialNB/BernoulliNB/ComplementNB/CategoricalNB `alpha=1.0`, `force_alpha=True`, `fit_prior=True`, `class_prior=None`; BernoulliNB `binarize=0.0`; ComplementNB `norm=False`; CategoricalNB `min_categories=None`.
- **D-03:** Shared NB math lives as **free functions in a `nb_common` module** (log-sum-exp normalize, empirical class-prior, argmax/argmin decode). The five estimators stay **fully independent structs** that call these helpers — **NO shared `NbBase` struct**, no inheritance coupling. Honors "five mutually-independent, parallel-buildable."
- **D-04:** None-meaning-default and polymorphic knobs typed Rust-natively (NOT sklearn stringly/None scalars):
  - `binarize: Option<f64>` — `None` disables binarization; `Some(t)` thresholds `x > t → 1`.
  - `priors` / `class_prior: Option<Vec<F>>` — `None` → empirical priors; `Some(..)` → user-supplied.
  - `min_categories` (CategoricalNB) — dedicated enum `MinCategories::{ Infer, Uniform(usize), PerFeature(Vec<usize>) }`.
- **D-04a (inherited):** Any genuinely-categorical knob → Rust enum with `TryFrom<&str>` (sklearn spelling) as single source of truth in `mlrs-algos`; PyO3 maps unknown values → `ValueError`.
- **D-05 (inherited split):** **Data-independent** knobs validate at **`build() -> Result<_, BuildError>`**; **data-dependent** checks at **`fit() -> AlgoError`**. PyO3 surfaces `BuildError` + enum `TryFrom` failures as `ValueError` at construction; `fit`-time `AlgoError` via existing `algo_err_to_py`.
  - **build():** `alpha >= 0`, `var_smoothing >= 0`, `min_categories` entries non-negative, `class_prior`/`priors` entries finite + non-negative.
  - **fit():** `class_prior`/`priors` **length == n_classes**, CategoricalNB non-negative-integer-encoded input, `n_features` agreement.
- **D-06:** sklearn **`force_alpha` parity**: when `force_alpha=False` and `alpha < 1e-10`, sklearn clips `alpha` to `1e-10` **and emits a warning**. Handled at **`build()`** (data-independent). Clip + warning must reproduce sklearn.
- **D-07:** Add **`predict_log_proba`** to the shared estimator trait surface (alongside `PredictProba`). Keep `PredictLabels` for `predict`. Add **`score`** via a shared helper (accuracy, mirroring sklearn `ClassifierMixin.score`).
- **D-08:** **ComplementNB's `argmin` decision rule stays internal** to its `PredictLabels` impl — same trait, different internal decision (note the sign). Do NOT special-case in trait or PyO3 layer.
- **D-09:** Builder method names + PY-06 Python-facing hyperparameter names **mirror sklearn exactly, per estimator**: `GaussianNB::builder().priors(..).var_smoothing(..)` (no `alpha`); the other four use `.class_prior(..).alpha(..)`. **Zero name-translation** in PyO3.
- **D-10:** **`partial_fit` is OUT OF SCOPE for Phase 11.** The five NB estimators implement `Fit` (not `PartialFit`).

### Claude's Discretion
- Exact builder method names beyond the sklearn-mirrored hyperparameters; the `BuildError` variant set; the precise field layout of each independent estimator struct — provided D-01…D-10 honored.
- The exact factoring of the `nb_common` free-function module (which helpers, signatures) — provided no shared base/config struct (D-03) and common math not 5×-duplicated.
- How the one-owner-per-(class,feature) GATHER kernel and CategoricalNB ragged `feature_log_prob_` are laid out on device — implementation detail (fixed only by ROADMAP success criteria + FEATURES.md parity notes).

### Deferred Ideas (OUT OF SCOPE)
- **NB `partial_fit`** — sklearn supports it for all five; PY-06 scopes `partial_fit` to IncrementalPCA/MBSGD only (D-10). Future: streaming NB (running mean/variance merge for GaussianNB, count accumulation for discrete variants).
- **Retrofitting Phases 4–9 low-arity estimators to builders** — deferred by Phase-10 D-02.
- **A shared `NbBase` struct / trait-object NB abstraction** — explicitly rejected (D-03 chose free functions).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NB-01 | Fit `GaussianNB` (per-class Gaussian likelihood with `var_smoothing`, log-sum-exp), `predict`/`predict_proba` ≤ tol | §"GaussianNB" math contract; §"GATHER idiom" (Pitfall 2); §"log-sum-exp + var_smoothing" (Pitfall 9); `nb_common::log_sum_exp_normalize` (D-03) |
| NB-02 | Fit `MultinomialNB` (multinomial likelihood, `alpha`), `predict`/`predict_proba` ≤ tol; sparse densified at ingress | §"MultinomialNB"; `feature_log_prob_` via class-grouped sum + GEMM joint-LL; ingress densify at PyO3 boundary (PROJ-02 precedent) |
| NB-03 | Fit `BernoulliNB` (`(1−x)·log(1−p)` non-occurrence term, `binarize`) ≤ tol | §"BernoulliNB"; the non-occurrence term + `binarize: Option<f64>` (D-04) |
| NB-04 | Fit `ComplementNB` (complement weights, argmin) ≤ tol | §"ComplementNB"; complement-count factoring + `norm` + internal `argmin` (D-08) |
| NB-05 | Fit `CategoricalNB` (per-feature categorical likelihood, `alpha`) on integer-encoded features ≤ tol | §"CategoricalNB ragged layout"; `MinCategories` enum (D-04); ragged `feature_log_prob_` host layout |
| PY-06 | All v2 estimators `#[pyclass]`-backed, sklearn-compatible `fit`/`predict`/`transform`/`score`, `get_params`/`set_params`, f32/f64 dispatch, GIL release, four per-backend wheels | §"PyO3 wrapping pattern"; `any_estimator!` macro + `py.detach` + `build_err_to_py`; final cross-cutting sign-off (estimator_checks re-triage) |
</phase_requirements>

## Summary

Phase 11 is the **lowest-risk family in v2.0**: five sklearn-compatible Naive Bayes classifiers built entirely as **host-orchestrated assembly over the validated v1 `reduce` prim plus host-side log/exp/log-sum-exp**. There is **no new CubeCL kernel and no new primitive** (ROADMAP "reductions-only closing bookend"). The five estimators are mutually independent and parallel-buildable; the only shared code is the `nb_common` free-function module (D-03).

The entire device surface NB needs already exists and is validated: `mlrs_backend::prims::reduce` (`row_reduce`/`column_reduce`/`sum` with `ScalarOp::Sum`), `gemm` (for the `X @ feature_log_prob_.T` joint-log-likelihood matvec, reused by MultinomialNB/BernoulliNB/ComplementNB exactly as `mbsgd_classifier.rs` uses it), and the host `f64_to_host`/`host_to_f64` casts. The "one-owner-per-(class,feature) GATHER kernel" requirement is satisfied **without writing a CubeCL kernel** — the v1 `row_reduce` prim already materializes per-row segments host-side and reduces each as a contiguous device buffer; class-conditional sufficient statistics are computed by host-grouping rows by class label (one owner per class) and reducing each group's feature columns. This is structurally a GATHER (each output cell reads its owned inputs), never a scatter-add (no two threads writing one cell, no atomics, no SharedMemory race), so it passes the `--features cpu` MLIR launch by construction.

**Primary recommendation:** Implement the five estimators as independent structs in a new `crates/mlrs-algos/src/naive_bayes/` module (mod.rs + 5 estimator files + `nb_common.rs`), copying the `mbsgd_classifier.rs` builder/`Default`/`build()->Result<_,BuildError>`/`classes_`-remap shape. Compute all class-conditional sufficient statistics by host-grouping rows by class and reducing column-wise over the v1 reduce prim (the GATHER idiom — no new kernel). Apply per-variant smoothing/denominator and the joint-LL in `f64` on the host (the device only does the bulk sums/GEMM), then normalize with a single `nb_common::log_sum_exp_normalize`. Wrap all five in PyO3 via the shipped `any_estimator!` macro + `py.detach` GIL-release pattern; extend the trait surface with `predict_log_proba` (D-07) and add a `nb_common::accuracy_score` helper. Gate every f64 oracle case with `skip_f64_with_log`; the f32-on-rocm band is documented per-variant; **exact predict labels are the hard gate for all five**.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Builder construction + data-independent validation | `mlrs-algos` (host) | — | `BuildError` raised before any data/device touch (D-05); pure host logic |
| `force_alpha` clip + warning | `mlrs-algos` builder `build()` | — | data-independent (D-06); `log::warn!` host-side |
| Class label inference / `classes_` remap | `mlrs-algos` `fit` (host) | — | host-side distinct-sort-dedup (mbsgd_classifier precedent), data-dependent |
| Class-conditional sufficient statistics (sums/counts/means/vars) | **`mlrs-backend` reduce prim (device)** | host grouping | the GATHER idiom: host groups rows by class (one owner per class), device reduces each group's columns |
| Joint log-likelihood `X @ feature_log_prob_.T` (discrete variants) | **`mlrs-backend` gemm (device)** | host bias-add | identical to mbsgd_classifier `decision_margin` GEMM matvec |
| Per-variant smoothing + denominators + log/exp of probs | `mlrs-algos` (host f64) | — | small `n_classes × n_features` tensors; host f64 avoids device log edge cases (Pitfall 9) |
| log-sum-exp normalize (`predict_proba`/`predict_log_proba`) | `mlrs-algos` `nb_common` (host f64) | — | row-wise over small `n_query × n_classes`; single log at end (KernelDensity precedent) |
| argmax/argmin label decode | `mlrs-algos` `nb_common` (host) | — | tiny per-row decision over `n_classes`; ComplementNB uses argmin (D-08) |
| `score` (accuracy) | `mlrs-algos` `nb_common` (host) | — | label-equality fraction (D-07) |
| Python surface (`fit`/`predict`/`predict_proba`/`predict_log_proba`/`score`, `get_params`/`set_params`, dtype dispatch, GIL release) | `mlrs-py` (`#[pyclass]`) | — | `any_estimator!` macro + `py.detach` (PY-06) |
| sparse densify (MultinomialNB) | `mlrs-py` ingress | — | densify at the Python ingress boundary (PROJ-02/Out-of-Scope §: no device CSR) |

## Standard Stack

This phase adds **zero new dependencies** (v2 milestone constraint: "v2 adds zero compute dependencies; no `cubek-random`, no pyo3 bump"). Everything is reuse of already-shipped in-repo crates.

### Core (all in-repo, already validated)
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `reduce` prim | `mlrs_backend::prims::reduce` | `row_reduce`/`column_reduce`/`sum` with `ScalarOp::Sum`; class-conditional sums/counts | The ONLY prim NB needs (ROADMAP); already host-segments rows (GATHER-shaped) `[VERIFIED: crates/mlrs-backend/src/prims/reduce.rs]` |
| `gemm` prim | `mlrs_backend::prims::gemm::gemm` | `X @ feature_log_prob_.T` joint-LL matvec for the 3 count-based variants | mbsgd_classifier uses identical signature for its decision margin `[VERIFIED: crates/mlrs-algos/src/linear/mbsgd_classifier.rs:506]` |
| Host casts | `mlrs_core::{f64_to_host, host_to_f64}` | f32↔f64 at the host/device boundary | Used by every estimator; KDE does all f64 math host-side this way `[VERIFIED: crates/mlrs-algos/src/density/kernel_density.rs:60]` |
| `DeviceArray` / `BufferPool` | `mlrs_backend::{device_array, pool}` | device-resident fitted state + buffer reuse | cross-cutting D-03; `release_into(pool)` on re-fit `[VERIFIED: kernel_density.rs:241]` |
| `Fit`/`PredictLabels`/`PredictProba` traits | `mlrs_algos::traits` | the uniform estimator surface to implement (+ new `predict_log_proba`) | `[VERIFIED: crates/mlrs-algos/src/traits.rs]` |
| `AlgoError`/`BuildError` | `mlrs_algos::error` | the two-tier (build vs fit) error contract to extend | `[VERIFIED: crates/mlrs-algos/src/error.rs]` |
| `any_estimator!` macro + `py.detach` | `mlrs_py::dispatch` / `estimators::linear` | dtype-dispatch enum + GIL-released device call for PyO3 | `[VERIFIED: crates/mlrs-py/src/dispatch.rs:91; estimators/linear.rs:995]` |

### Supporting (reference implementations to copy)
| Reference file | What to copy | When |
|----------------|--------------|------|
| `linear/mbsgd_classifier.rs` | builder()/`Default`/`build()->Result<_,BuildError>`; `classes_` distinct-sort-dedup; `PredictLabels`+`PredictProba`; GEMM matvec `decision_margin`; `release_into` re-fit | every NB struct (the closest analog) `[VERIFIED]` |
| `density/kernel_density.rs` | `KdKernel`/`BandwidthSpec` enum + `Option` precedent for `MinCategories`; **host-side log-sum-exp with single terminal `log`** (Pitfall 3 — never `±∞` on device); host f64 math | `MinCategories` enum + `nb_common` log-sum-exp `[VERIFIED]` |
| `linear/linear_svc.rs` | second builder example (`build()` validates `C>0` + loss family) | `BuildError` variant pattern `[VERIFIED]` |
| `estimators/linear.rs` `PyMBSGDClassifier` | full `#[pyclass]` builder wrapper: `Unfit` arm holding sklearn-named knobs, `TryFrom` enum parse → `build_err_to_py`, `py.detach` device call, dtype dispatch | every Py NB wrapper `[VERIFIED]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Host-grouped reduce (GATHER) | A new scatter-add CubeCL kernel keyed by class | REJECTED by ROADMAP success criterion #1 ("NO scatter-add") AND by cubecl-cpu (no atomics/SharedMemory races pass the cpu launch). The reduce-prim path is mandated and already validated. |
| Host f64 joint-LL assembly | A device log-sum-exp kernel | The KDE precedent (host f64, single terminal log) is the established Pitfall-3/9-safe pattern; `n_query × n_classes` is tiny so no device launch is warranted |
| `Option<Vec<F>>` priors | sklearn None-overloaded scalar | D-04 locks the Rust-native typing |
| Ragged `Vec<Vec<F>>` for CategoricalNB `feature_log_prob_` | a padded single tensor | sklearn keeps it ragged (variable category count per feature); FEATURES.md ⚠ "ragged list — not a single tensor". Pad only via `MinCategories` |

**Installation:** None — no new crates. Module wiring only:
- Add `pub mod naive_bayes;` to `crates/mlrs-algos/src/lib.rs` (alphabetical after `linear`).
- Add `predict_log_proba` to `crates/mlrs-algos/src/traits.rs` (`PredictLogProba` trait, or default method on a new trait mirroring `PredictProba`).
- Register five `#[pyclass]` structs in `crates/mlrs-py/src/lib.rs` (`m.add_class::<PyGaussianNB>()?;` etc.) and a new `crates/mlrs-py/src/estimators/naive_bayes.rs`.

**Version verification:** N/A — no external packages installed this phase. The oracle reference is `scikit-learn >=1.6` (already pinned in the repo's oracle venv; FEATURES.md cites sklearn 1.9.0 source). `[VERIFIED: REQUIREMENTS.md "v2 adds zero compute dependencies"]`

## Package Legitimacy Audit

**N/A — this phase installs no external packages.** Phase 11 is pure Rust assembly over already-shipped in-repo crates (`mlrs-core`, `mlrs-backend`, `mlrs-kernels`, `mlrs-algos`, `mlrs-py`) plus PyO3 0.28 (already a dependency, no bump per v2 Out-of-Scope). No npm/PyPI/crates install steps. The oracle generator `scripts/gen_oracle.py` uses `numpy`/`scipy`/`scikit-learn` in a throwaway `/tmp` venv (build-time only, blobs committed; never run in CI) — these are not phase runtime dependencies.

## Architecture Patterns

### System Architecture Diagram

```
                 Python user code (sklearn-compatible)
   est = GaussianNB(var_smoothing=1e-9); est.fit(X, y); est.predict_proba(Xq)
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  mlrs-py  (#[pyclass] PyGaussianNB … PyCategoricalNB)             │
   │  • Unfit arm holds sklearn-named knobs verbatim                   │
   │  • MultinomialNB: sparse → densify at ingress                    │
   │  • float_dtype(X) → F32|F64 dispatch                              │
   │  • py.detach(|| { lock_pool(); … })   ← GIL released (PY-06)     │
   │  • build_err_to_py / algo_err_to_py → ValueError / RuntimeError  │
   └──────────────────────────────────────────────────────────────────┘
                              │  AnyGaussianNB::{Unfit, F32(_), F64(_)}
                              ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  mlrs-algos::naive_bayes  (5 independent structs)                 │
   │                                                                    │
   │  builder() ─Default(sklearn defaults D-02)→ build() ─validate──┐  │
   │     data-INDEPENDENT (alpha≥0, var_smoothing≥0, force_alpha     │  │
   │     clip+warn) → Result<Estimator, BuildError>                 │  │
   │                                                            ◄────┘  │
   │  fit(pool, X, y, shape):                                           │
   │   1. host: classes_ = sort∘dedup(y)  (mbsgd precedent)            │
   │   2. host: data-DEPENDENT validate (prior len==n_classes, …)     │
   │   3. GATHER sufficient stats  ──────────────┐                     │
   │   4. host f64: per-variant smoothing+denom  │                     │
   │      → theta_/var_/feature_log_prob_/weights│ (device-resident)   │
   │                                              │                     │
   │  predict_*/score:                            │                     │
   │   joint_ll = class_log_prior_ + X@flp_.T ────┤ (GEMM, discrete)   │
   │      or  −0.5·Σ[log(2πv)+(x−μ)²/v]  (Gaussian)                    │
   │   nb_common::log_sum_exp_normalize(joint_ll) → proba (rows sum 1) │
   │   argmax (or argmin for ComplementNB, D-08) → labels             │
   └──────────────────────────────────────────────────────────────────┘
            │ class-conditional sums/counts        │ X @ M.T
            ▼ (the GATHER)                          ▼
   ┌──────────────────────────────────────┐  ┌────────────────────────┐
   │ mlrs-backend::reduce  (v1, validated)│  │ mlrs-backend::gemm (v1)│
   │ row/column_reduce, ScalarOp::Sum     │  │ matvec, no new kernel  │
   │ host-segments per (class-group, col) │  └────────────────────────┘
   │ → ONE owner per (class,feature)      │
   │ → NO scatter-add, NO atomics         │
   │ → passes --features cpu MLIR launch  │
   └──────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/mlrs-algos/src/
├── traits.rs                    # + PredictLogProba trait (D-07)
├── error.rs                     # + NB BuildError/AlgoError variants
├── naive_bayes/
│   ├── mod.rs                   # pub mod re-exports; module-level doc
│   ├── nb_common.rs             # FREE FUNCTIONS only (D-03), no struct:
│   │                            #   log_sum_exp_normalize, empirical_class_log_prior,
│   │                            #   argmax_decode, argmin_decode, accuracy_score,
│   │                            #   class_grouped_sum (the GATHER helper)
│   ├── gaussian_nb.rs           # GaussianNB struct + builder + Fit/PredictLabels/PredictProba/PredictLogProba
│   ├── multinomial_nb.rs        # MultinomialNB
│   ├── bernoulli_nb.rs          # BernoulliNB  (binarize: Option<f64>)
│   ├── complement_nb.rs         # ComplementNB (norm, internal argmin)
│   └── categorical_nb.rs        # CategoricalNB (MinCategories enum, ragged feature_log_prob_)

crates/mlrs-algos/tests/
├── gaussian_nb_test.rs          # oracle: exact labels (hard) + proba band + default-matches-sklearn + build-rejects
├── multinomial_nb_test.rs
├── bernoulli_nb_test.rs
├── complement_nb_test.rs
└── categorical_nb_test.rs

crates/mlrs-py/src/estimators/
└── naive_bayes.rs               # 5 #[pyclass] wrappers via any_estimator!

scripts/gen_oracle.py            # + gen_gaussian_nb / gen_multinomial_nb / … (committed .npz blobs)
tests/fixtures/                  # + *_nb_{f32,f64}_seed42.npz
```

### Pattern 1: The one-owner-per-(class,feature) GATHER (no new kernel)
**What:** Class-conditional sufficient statistics (per-class feature sums, counts, sum-of-squares) computed so each output cell `(class c, feature j)` is the reduction of exactly the rows whose label is `c` — one owner, never a contended write.
**When to use:** Every NB variant's `fit` (the only place device sums are needed).
**How (host orchestration over the v1 reduce prim — NO `#[cube]` code):**
1. Host materializes `y` once, builds `classes_` (sort∘dedup, mbsgd precedent) and a per-class row-index list.
2. For each class `c`: gather that class's rows into a contiguous `n_c × n_features` device buffer (host slice → `DeviceArray::from_host`, the same per-segment materialization `row_reduce` already does internally).
3. `column_reduce(ScalarOp::Sum)` over that buffer → length-`n_features` per-class feature sum (the GATHER output row for class `c`). Count `n_c` is the row count; sum-of-squares for GaussianNB variance uses a squared copy or `ScalarOp::SumSq`-style pass.
4. Release each per-class scratch buffer back into the pool (WR-07 re-fit reuse).

This is structurally identical to how `row_reduce`/`column_reduce` already work (`[VERIFIED: reduce.rs:202-219]` — they host-slice each segment, upload it, reduce on device, release). **No SharedMemory race, no atomics, no `F::INFINITY`, no shift-loop** → passes `--features cpu` (Pitfall 2 / [[cubecl-cpu-no-shared-memory]]).
**Example (signature in `nb_common`):**
```rust
// Source: reduce-prim composition (reduce.rs column_reduce precedent)
/// Per-class column sums: returns an n_classes × n_features host matrix where
/// row c, col j = Σ_{i : y_i == classes_[c]} x[i, j]. One owner per (class,feature)
/// — a GATHER, never a scatter-add. Composes column_reduce over per-class
/// row-gathered buffers; cpu-MLIR-safe.
fn class_grouped_sum<F: Float + CubeElement + Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),          // (n_samples, n_features)
    class_of_row: &[usize],          // length n_samples, index into classes_
    n_classes: usize,
) -> Vec<Vec<f64>>;                   // n_classes × n_features, f64 host
```

### Pattern 2: Discrete-variant joint log-likelihood via GEMM
**What:** `joint_ll = class_log_prior_[c] + (X @ feature_log_prob_.T)[i, c]` for MultinomialNB/BernoulliNB/ComplementNB.
**When:** `predict_labels`/`predict_proba`/`predict_log_proba` of the three count-based variants.
**How:** Identical to `mbsgd_classifier::decision_margin` — `gemm(X (n×d), flp (n_classes×d) transposed, …)` → `n × n_classes`, then host-add the per-class log-prior bias. BernoulliNB additionally adds the per-class constant `Σ_j log(1−p_cj)` and uses `flp_cj = log p_cj − log(1−p_cj)` so the `(1−x)·log(1−p)` term folds into a constant + the GEMM (sklearn `_joint_log_likelihood`).
**Example:**
```rust
// Source: mbsgd_classifier.rs:506 (gemm matvec), generalized to n_classes columns
let raw = gemm::<F>(pool, x, (n_query, n_features), flp_dev, (n_features, n_classes),
                    /*transpose_a*/ false, /*transpose_b*/ true, None)?;
// host: joint_ll[i][c] = class_log_prior[c] (+ bernoulli_neg_const[c]) + raw[i][c]
```

### Pattern 3: log-sum-exp normalize (single terminal log, host f64)
**What:** `predict_proba` = `exp(joint_ll − logsumexp_c(joint_ll))`; `predict_log_proba` = `joint_ll − logsumexp`.
**When:** All five variants (the rows-sum-to-1 gate, success criterion #3).
**How:** Per query row: `m = max_c joint_ll[c]; lse = m + log(Σ_c exp(joint_ll[c] − m)); proba[c] = exp(joint_ll[c] − lse)`. All in host f64 (the joint_ll is a tiny `n_query × n_classes` host matrix already). This mirrors the KDE pattern: the single `log` applied once at the end, never `±∞` mid-pipeline (Pitfall 3). `[CITED: KernelDensity score_samples §"Linear-domain log-sum-exp"]`
**Example:**
```rust
// Source: nb_common (KDE log-sum-exp precedent, kernel_density.rs:342)
fn log_sum_exp_normalize(joint_ll: &[f64], n_classes: usize) -> (Vec<f64> /*proba*/, Vec<f64> /*log_proba*/);
```

### Anti-Patterns to Avoid
- **A scatter-add / atomic class-accumulation kernel** — forbidden by ROADMAP #1 and fails the cpu launch. Use the host-grouped reduce GATHER (Pattern 1).
- **A device-side log-sum-exp with `F::INFINITY` sentinels or descending-shift reduction** — triggers the cubecl-cpu MLIR panic (Pitfall 2). Do the log-sum-exp host-side in f64 (Pattern 3).
- **A shared `NbBase` struct or trait object** — D-03 forbids it; share via free functions only.
- **Per-class variance for `var_smoothing`** — GaussianNB's `epsilon_` is `var_smoothing · max_j(Var(X[:,j]))` over the **whole dataset** (global, not per-class). Common bug; FEATURES.md ⚠.
- **Using `alpha·1` as the multinomial denominator** — must be `alpha · n_features` (MultinomialNB) / `2·alpha` (BernoulliNB) / `alpha · n_categories_j` (CategoricalNB). FEATURES.md ⚠.
- **Copying MultinomialNB into ComplementNB** — CNB uses complement counts (all classes except c), optional L1 `norm`, and **argmin** (note the sign). FEATURES.md ⚠.
- **Treating CategoricalNB `feature_log_prob_` as one tensor** — it is a ragged list (one matrix per feature, variable category count). FEATURES.md ⚠.
- **Translating hyperparameter names in PyO3** — D-09: mirror sklearn names exactly per estimator (GaussianNB has no `alpha`; others have no `var_smoothing`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Class-conditional sums/counts | A scatter-add CubeCL kernel | `reduce::column_reduce`/`row_reduce` over host-grouped rows | Validated v1 prim; ROADMAP mandates reductions-only; cpu-launch-safe |
| `X @ feature_log_prob_.T` | A bespoke matvec kernel | `gemm` prim (mbsgd precedent) | Already validated; transpose_b flag handles the `.T` |
| log-sum-exp | A device reduction with infinity sentinels | host f64 max-shift + single terminal log (KDE precedent) | Pitfall 3/9; tensor is tiny; avoids cubecl-cpu panic |
| f32↔f64 boundary casts | manual transmute | `mlrs_core::{f64_to_host, host_to_f64}` | Used project-wide; mbsgd/KDE precedent |
| Builder + split validation | a `new(positional)` ctor | copy `mbsgd_classifier.rs` builder/`Default`/`build()` | D-01/D-05 locked; reference is shipped |
| dtype dispatch + GIL release in PyO3 | hand-written enum | `any_estimator!` macro + `py.detach` | PY-06; v2 adds zero binding infra |
| `BuildError`→`ValueError` mapping | a new mapper | existing `build_err_to_py` / `algo_err_to_py` | single-site mapping (D-09) |
| sparse input (MultinomialNB) | a device CSR path | densify at PyO3 ingress | v2 Out-of-Scope: "accepts sparse by densifying at ingress" (PROJ-02 precedent) |

**Key insight:** NB has **no device math of its own** beyond bulk sums (reduce) and one matvec (gemm), both already validated. The per-variant intelligence (smoothing, denominators, the Bernoulli non-occurrence term, complement weighting, ragged categorical lookup, log-sum-exp) is all small-tensor host f64 arithmetic. Building any new device kernel here is both unnecessary and a cpu-launch hazard.

## Runtime State Inventory

> Not a rename/refactor/migration phase — greenfield estimator additions. Section omitted per template.

## Common Pitfalls

### Pitfall 1: GATHER mis-expressed as scatter-add (ROADMAP #1, cubecl-cpu)
**What goes wrong:** Implementing class-conditional accumulation as "for each row, atomically add to its class's accumulator" — multiple threads write one `(class,feature)` cell.
**Why it happens:** It is the textbook GPU histogram pattern, but cubecl-cpu has no atomics/SharedMemory-safe path and ROADMAP forbids it.
**How to avoid:** Host-group rows by class first; reduce each group's columns with the v1 reduce prim (one owner per cell). Verify with `cargo test --features cpu` (launch, not just build).
**Warning signs:** Any new `#[cube]` kernel in this phase; an MLIR `failed to run pass` panic at launch; use of `Atomic`/`SharedMemory`/`F::INFINITY` in NB code.

### Pitfall 2: cpu-MLIR launch failure from a SharedMemory/infinity kernel
**What goes wrong:** A log-sum-exp or argmax kernel using `F::INFINITY` init + a mutable-bool flag + a descending-shift reduction compiles but panics at `--features cpu` launch.
**Why it happens:** Documented cubecl-cpu 0.10 limitation ([[cubecl-cpu-no-shared-memory]]).
**How to avoid:** NB writes NO new device kernel. log-sum-exp/argmax/argmin are host f64. The only device ops are the already-validated reduce (Shared path, cpu-safe) and gemm.
**Warning signs:** Same as Pitfall 1.

### Pitfall 3: GaussianNB `epsilon_` from per-class instead of global variance
**What goes wrong:** Computing `var_smoothing · Var` per class rather than `var_smoothing · max_j(Var(X[:,j]))` over the whole dataset (ddof=0 population variance).
**Why it happens:** It reads naturally as a per-class quantity; sklearn computes it globally once.
**How to avoid:** Compute `epsilon_` ONCE from the full `X` column variances (ddof=0), take the max over features, add to every class's `var_`. FEATURES.md ⚠. `[CITED: FEATURES.md §GaussianNB]`
**Warning signs:** `var_` differs from sklearn by a constant per-class offset; proba band fails only on the smallest-variance feature.

### Pitfall 4: Wrong per-variant smoothing denominator
**What goes wrong:** Using `alpha·1` everywhere. The denominators differ: MultinomialNB `Σ_j count[c,j] + alpha·n_features`; BernoulliNB `class_count[c] + 2·alpha`; CategoricalNB `class_count[c] + alpha·n_categories_j`.
**Why it happens:** The variants look similar; the denominator is the distinguishing detail.
**How to avoid:** Implement each `feature_log_prob_` formula verbatim from FEATURES.md §per-variant. `[CITED: FEATURES.md §Multinomial/Bernoulli/Categorical]`
**Warning signs:** proba off by a smoothing-scale factor; mismatch grows with `alpha`.

### Pitfall 5: BernoulliNB missing the `(1−x)·log(1−p)` non-occurrence term
**What goes wrong:** Treating Bernoulli like Multinomial (only the occurrence term).
**How to avoid:** `LL = class_log_prior + Σ_j[ x_j·log p_cj + (1−x_j)·log(1−p_cj) ]`. Fold into GEMM via `flp = log p − log(1−p)` plus a per-class constant `Σ_j log(1−p_cj)` (sklearn's `neg_prob`). Also apply `binarize` (D-04 `Option<f64>`: `None` → assume binary, `Some(t)` → `x>t`). `[CITED: FEATURES.md §Bernoulli]`
**Warning signs:** Bernoulli proba matches Multinomial but not sklearn.

### Pitfall 6: ComplementNB sign / argmin / complement weighting
**What goes wrong:** Copying MNB: using class counts (not complement counts), forgetting optional L1 `norm`, or using argmax instead of **argmin**.
**How to avoid:** `weights = log((complement_count + alpha)/(complement_count.sum() + alpha·n_features))` where complement_count[c,j] = Σ_{c'≠c} feature_count[c',j]; if `norm=True` L1-normalize weights; decision = **argmin** of `X @ weights` (D-08, internal to CNB's `PredictLabels`). `[CITED: FEATURES.md §Complement]`
**Warning signs:** Labels are the exact complement of sklearn's (sign flip); norm-default mismatch.

### Pitfall 7: CategoricalNB ragged layout + unseen categories + min_categories
**What goes wrong:** Forcing a single tensor; not handling a predict-time category index ≥ training categories; ignoring `min_categories` padding.
**How to avoid:** `feature_log_prob_` is `Vec<Vec<f64>>`-shaped (per feature j: `n_classes × n_categories_j`). `n_categories_j = max(observed+1, min_categories_j)` (`MinCategories` enum, D-04). Validate inputs are non-negative integers at `fit` (D-05, `AlgoError`). At predict, an unseen category maps to the smoothed prob (`log(alpha / denom)`); guard the lookup index. `[CITED: FEATURES.md §Categorical]`
**Warning signs:** Index-out-of-bounds at predict; mismatch when a query has a category absent from training.

### Pitfall 8: f64 oracle run on rocm without the skip gate
**What goes wrong:** A GaussianNB f64 oracle test fails on rocm (no f64 / `SHADER_F64`).
**How to avoid:** Every f64 oracle test calls `capability::skip_f64_with_log()` and `return`s early when true (mbsgd_classifier precedent). f32 runs at a documented band; the **f32-on-rocm band for GaussianNB log-proba** is the one variant flagged for an explicit tolerance band (log-proba magnifies f32 round-off). `[VERIFIED: capability.rs:147; mbsgd_classifier_test.rs]`
**Warning signs:** rocm CI red on an f64 case; a GaussianNB proba band too tight for f32.

### Pitfall 9: log-sum-exp underflow / rows not summing to 1
**What goes wrong:** Computing `exp(joint_ll)` directly underflows to 0 for very negative LLs, so the row doesn't sum to 1.
**How to avoid:** Subtract the per-row max before exp (Pattern 3); single terminal log. Assert each `predict_proba` row sums to 1 within tolerance (success criterion #3). `[CITED: KernelDensity §Linear-domain log-sum-exp]`
**Warning signs:** A proba row of all-zeros or summing to ≠1.

## Code Examples

### Builder + split validation + force_alpha clip (a discrete variant)
```rust
// Source: mbsgd_classifier.rs builder/build() pattern, adapted for NB D-05/D-06
impl MultinomialNBBuilder {
    pub fn alpha(mut self, alpha: f64) -> Self { self.alpha = alpha; self }
    pub fn force_alpha(mut self, f: bool) -> Self { self.force_alpha = f; self }
    pub fn fit_prior(mut self, f: bool) -> Self { self.fit_prior = f; self }
    pub fn class_prior(mut self, p: Option<Vec<f64>>) -> Self { self.class_prior = p; self }

    pub fn build<F: Float + CubeElement + Pod>(self) -> Result<MultinomialNB<F>, BuildError> {
        // data-INDEPENDENT (D-05): alpha >= 0
        if !(self.alpha >= 0.0) {
            return Err(BuildError::InvalidAlpha { estimator: "multinomial_nb", alpha: self.alpha });
        }
        // D-06 force_alpha parity: clip + warn (data-independent)
        let alpha = if !self.force_alpha && self.alpha < 1e-10 {
            log::warn!("alpha too small, setting alpha=1e-10. Use force_alpha=True to keep alpha unchanged.");
            1e-10
        } else { self.alpha };
        // class_prior ENTRIES finite + non-negative here; LENGTH==n_classes deferred to fit (D-05)
        if let Some(p) = &self.class_prior {
            if p.iter().any(|&v| !(v.is_finite() && v >= 0.0)) {
                return Err(BuildError::InvalidClassPrior { estimator: "multinomial_nb" });
            }
        }
        Ok(MultinomialNB { alpha, fit_prior: self.fit_prior, class_prior: self.class_prior,
                            classes_: Vec::new(), /* device-resident fitted = None */ .. })
    }
}
```

### PyO3 wrapper (mirrors PyMBSGDClassifier; D-09 sklearn names, GIL release)
```rust
// Source: estimators/linear.rs PyMBSGDClassifier (any_estimator! + py.detach)
crate::any_estimator! {
    any: AnyGaussianNB,
    algo: mlrs_algos::naive_bayes::gaussian_nb::GaussianNB,
    unfit: { var_smoothing: f64, priors: Option<Vec<f64>> },   // sklearn names (D-09): NO alpha
}

#[pymethods]
impl PyGaussianNB {
    #[new]
    #[pyo3(signature = (var_smoothing = 1e-9, priors = None))]   // sklearn defaults (D-02)
    fn new(var_smoothing: f64, priors: Option<Vec<f64>>) -> Self { /* Unfit arm */ }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>,
           rows: usize, cols: usize) -> PyResult<()> {
        // … float_dtype dispatch …
        let fitted = py.detach(|| -> PyResult<AnyGaussianNB> {     // GIL released (PY-06)
            let mut pool = crate::lock_pool();
            // builder().var_smoothing(..).priors(..).build().map_err(build_err_to_py)?
            //   .fit(&mut pool, &xd, Some(&yd), (rows, cols)).map_err(algo_err_to_py)?
        })?;
        Ok(())
    }
    // predict / predict_proba / predict_log_proba / score / get_params / set_params …
}
```

### f64 oracle test skeleton (skip_f64 + exact-labels hard gate)
```rust
// Source: mbsgd_classifier_test.rs exact_labels pattern
#[test]
fn exact_labels() {                       // f64 case
    if capability::skip_f64_with_log() { return; }   // rocm skips, cpu runs
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    let case = load_npz(fixture("gaussian_nb_f64_seed42.npz")).unwrap();
    let predict_ref: Vec<i32> = case.expect_f64("predict").iter().map(|&v| v.round() as i32).collect();
    let (labels, _proba) = fit_gaussian::<f64>(&case);
    assert_eq!(labels, predict_ref, "GaussianNB f64 exact predict labels (HARD gate)");
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sklearn `force_alpha` default `'warn'` | `force_alpha=True` default | sklearn 1.4 | D-02 default is `True`; D-06 clip+warn only when explicitly `False` and `alpha<1e-10` |
| Hand-rolled per-estimator PyO3 enum | `any_estimator!` macro + `py.detach` | v2 Phase 6→10 | Zero new binding infra (PY-06) |
| Device scatter-add histograms | host-grouped reduce GATHER | mlrs v2 (cubecl-cpu constraint) | ROADMAP #1 mandates it |

**Deprecated/outdated:**
- sklearn `BernoulliNB(binarize=...)` accepting `None` to skip — mlrs models this as `Option<f64>` (D-04), not a None-overloaded float.
- A dedicated v1 Python phase for bindings — superseded by incremental per-phase wrapping (v2 reuses the shipped layer; ROADMAP PY-06 placement note).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The v1 `gemm` `transpose_b=true` path is available and validated for the `X @ feature_log_prob_.T` matvec with `n_classes > 1` columns | Pattern 2 / Don't-Hand-Roll | LOW — mbsgd uses gemm with a 1-column operand; a multi-column transpose-B is the same prim, but the planner should confirm `gemm` accepts `(n_features, n_classes)` B with `transpose_b` (Wave 0 check). If not, transpose `flp` host-side before upload. |
| A2 | `force_alpha=False & alpha<1e-10` clip threshold is `1e-10` and the warning text matches sklearn closely enough for parity (no proba dependence on the exact string) | force_alpha (D-06) | LOW — proba parity depends only on the clipped numeric `1e-10`, not the warning text. Verify the threshold against the pinned sklearn version when generating the oracle. |
| A3 | CategoricalNB unseen-category-at-predict maps to the smoothed `log(alpha/denom)` (sklearn clips the index / treats as count 0) | Pitfall 7 | MEDIUM — sklearn 1.6+ raises/handles unseen categories in a specific way; the planner should pin the oracle fixture to a case WITHOUT unseen categories first (exact-label gate), then add an unseen-category fixture and match sklearn's documented behavior. |
| A4 | The f32-on-rocm band for GaussianNB log-proba is wider than the other four variants (log-proba magnifies f32 round-off) but still passes EXACT labels | Pitfall 8 / Validation | MEDIUM — band width is empirical; the planner sets it from the actual f32 vs f64 residual at fixture-gen time (mbsgd precedent re-pinned bands after measuring). |
| A5 | `ScalarOp` exposes a sum-of-squares per-axis path (or one is composable via a squared host copy) for GaussianNB variance | Pattern 1 | LOW — full-array `l2_norm`/`SumSq` exist (`reduce.rs`); per-row/col `SumSq` may need a squared-copy then `Sum`. Either works; planner picks at Wave 0. |

**All other claims** (builder shape, trait surface, error tiers, PyO3 macro, reduce/gemm signatures, skip_f64 gate, GATHER feasibility, per-variant math) are `[VERIFIED]` against in-repo source or `[CITED: FEATURES.md]` (the pinned math contract). 

## Open Questions

1. **Plan granularity: one plan per variant, or grouped?**
   - What we know: the five are mutually independent and parallel-buildable (D-03, ROADMAP). PY-06 is a final cross-cutting sign-off.
   - What's unclear: whether to split as 5 estimator plans + 1 PY-06 plan, or group (e.g., Gaussian alone; the 3 count-based together since they share the GEMM joint-LL; Categorical alone; PY-06).
   - Recommendation: Wave structure — Wave 0 (traits.rs `predict_log_proba` + `nb_common` free functions + error variants + oracle fixtures), then parallel estimator plans, then a final PY-06 plan (wrap all five + estimator_checks re-triage across the full v2 surface). Grouping the 3 count-based variants is reasonable (shared GEMM/log-sum-exp path) but D-03 explicitly values independent buildability — keep structs separate even if planned together.

2. **`score` placement (D-07).**
   - What we know: D-07 adds `score` "via a shared helper (accuracy)".
   - What's unclear: whether `score` is a trait method or a free function in `nb_common` called by each PyO3 wrapper.
   - Recommendation: `nb_common::accuracy_score(predicted_labels, y_true) -> f64` free function (D-03 — no shared struct/trait state); each `#[pyclass]` calls it after `predict_labels`. Mirrors sklearn `ClassifierMixin.score` without coupling the five structs.

## Environment Availability

> Skipped — Phase 11 is a code-only Rust phase with no external runtime dependencies. The compute stack (`mlrs-backend` reduce/gemm prims, CubeCL cpu/rocm runtimes) is already present and validated through Phase 10. Oracle fixtures are committed `.npz` blobs (no Python in the test loop); `gen_oracle.py` regeneration needs a throwaway `/tmp` venv with numpy/scipy/scikit-learn (build-time only, not a runtime/CI dependency — see project memory `oracle-fixture-regen-needs-venv`).

## Validation Architecture

> nyquist_validation is enabled (`.planning/config.json workflow.nyquist_validation: true`). This section drives VALIDATION.md.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `mlrs_core::{load_npz, OracleCase}` oracle harness; sklearn `.npz` fixtures |
| Config file | none — `cargo test` per crate; fixtures in `tests/fixtures/*.npz` (committed) |
| Quick run command | `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test` (per-variant, targeted) |
| Full suite command | `cargo test --features cpu -p mlrs-algos` (NB tests) + `cargo test --features rocm -p mlrs-algos` (f32 gate); PyO3: `cargo test -p mlrs-py` |
| Oracle regen (build-time only) | `/tmp/oracle-venv/bin/python scripts/gen_oracle.py` (numpy/scipy/scikit-learn; blobs committed, CI never runs it) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NB-01 | GaussianNB exact predict labels (HARD gate) | oracle unit | `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test exact_labels` | ❌ Wave 0 |
| NB-01 | GaussianNB predict_proba band + rows-sum-to-1 | oracle unit | `… gaussian_nb_test proba_band` | ❌ Wave 0 |
| NB-01 | GaussianNB `builder().build()` == sklearn default (var_smoothing=1e-9) | unit | `… gaussian_nb_test default_matches_sklearn` | ❌ Wave 0 |
| NB-01 | build() rejects `var_smoothing < 0` (BuildError) | unit | `… gaussian_nb_test build_rejects_bad_var_smoothing` | ❌ Wave 0 |
| NB-02 | MultinomialNB exact labels + proba band; densify path | oracle unit | `… multinomial_nb_test exact_labels` / `proba_band` | ❌ Wave 0 |
| NB-02 | build() rejects `alpha < 0`; force_alpha clip+warn | unit | `… multinomial_nb_test build_rejects_bad_alpha` / `force_alpha_clip` | ❌ Wave 0 |
| NB-03 | BernoulliNB exact labels (incl. `(1−x)log(1−p)` term, binarize) | oracle unit | `… bernoulli_nb_test exact_labels` | ❌ Wave 0 |
| NB-03 | BernoulliNB `binarize=None` (assume-binary) path | oracle unit | `… bernoulli_nb_test binarize_none` | ❌ Wave 0 |
| NB-04 | ComplementNB exact labels (argmin, complement weights) | oracle unit | `… complement_nb_test exact_labels` | ❌ Wave 0 |
| NB-04 | ComplementNB `norm=True` weight L1-normalize | oracle unit | `… complement_nb_test norm_true` | ❌ Wave 0 |
| NB-05 | CategoricalNB exact labels (ragged feature_log_prob_) | oracle unit | `… categorical_nb_test exact_labels` | ❌ Wave 0 |
| NB-05 | CategoricalNB `min_categories` padding (MinCategories enum) | oracle unit | `… categorical_nb_test min_categories` | ❌ Wave 0 |
| NB-05 | fit() rejects negative / non-integer categorical input (AlgoError) | unit | `… categorical_nb_test fit_rejects_bad_input` | ❌ Wave 0 |
| (all) | every f64 oracle case skips on rocm via `skip_f64_with_log` | gate | embedded in each `exact_labels` (f64) test | ❌ Wave 0 |
| (all) | GATHER kernel path passes `--features cpu` launch | gate | any oracle test compiled+run with `--features cpu` (launch witness) | ❌ Wave 0 |
| (all) | PoolStats memory gate per estimator (no leak across re-fit) | unit | `… <variant>_test refit_releases_buffers` (PoolStats live_bytes assert) | ❌ Wave 0 |
| PY-06 | each `#[pyclass]` instantiates + fit/predict/predict_proba/predict_log_proba/score round-trips | smoke | `cargo test -p mlrs-py --test pyclass_smoke_test` (extend) | ⚠️ extend existing |
| PY-06 | get_params/set_params with sklearn-named knobs; f32/f64 dispatch; GIL release | smoke | `cargo test -p mlrs-py` | ⚠️ extend existing |
| PY-06 | estimator_checks re-triaged across full v2 surface | manual/integration | sklearn `check_estimator` (Python, end-of-phase) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --features cpu -p mlrs-algos --test <variant>_nb_test` (the variant being edited — ~seconds, NOT the full mlrs-algos suite which is ~6 min, project memory `backend-test-suite-slow`).
- **Per wave merge:** `cargo test --features cpu -p mlrs-algos --test gaussian_nb_test --test multinomial_nb_test … ` (all five NB tests, targeted) + `cargo test -p mlrs-py` smoke. Background the full backend suite if needed.
- **Phase gate:** all five NB oracle tests green on `--features cpu` (f64) AND `--features rocm` (f32, f64 skipped-with-log); PY-06 smoke green; exact-labels hard gate green for all five; every `predict_proba` row sums to 1; PoolStats no-leak per estimator. Then `/gsd-verify-work`.

### Coverage strategy across the 5 variants × 2 dtypes
- **Both dtypes per variant:** each variant gets `*_f32_seed42.npz` and `*_f64_seed42.npz` fixtures (gen_oracle.py `dtype` param, existing convention). f64 tests gated by `skip_f64_with_log`; f32 tests run a documented band.
- **Exact labels = hard gate (no band) for all five** — integers, the primary correctness witness. The proba band is secondary (bounds last-bit drift).
- **f32-on-rocm band:** GaussianNB log-proba gets the widest documented band (A4); the four discrete variants are integer-count-based and band tighter.
- **One small geometry** per variant (mirror SGD's `40×4`, `8` query rows) — well-separated classes so exact labels are unambiguous (the `_sgd_blobs` class-blob generator is directly reusable for the continuous variants; the discrete variants need integer-count `X` and the categorical variant needs integer-encoded features — new small generators).

### Wave 0 Gaps
- [ ] `crates/mlrs-algos/src/traits.rs` — add `PredictLogProba` trait (D-07)
- [ ] `crates/mlrs-algos/src/naive_bayes/nb_common.rs` — free functions: `log_sum_exp_normalize`, `empirical_class_log_prior`, `argmax_decode`, `argmin_decode`, `accuracy_score`, `class_grouped_sum` (the GATHER helper)
- [ ] `crates/mlrs-algos/src/error.rs` — NB `BuildError` variants (e.g. `InvalidVarSmoothing`, `InvalidClassPrior`, reuse `InvalidAlpha`) + `AlgoError` variants (e.g. `InvalidCategoricalInput`, prior-length mismatch via `InvalidLabels`/new)
- [ ] `scripts/gen_oracle.py` — `gen_gaussian_nb`/`gen_multinomial_nb`/`gen_bernoulli_nb`/`gen_complement_nb`/`gen_categorical_nb` + integer-count / categorical-encoded data generators; commit `tests/fixtures/*_nb_{f32,f64}_seed42.npz`
- [ ] `crates/mlrs-algos/tests/{gaussian,multinomial,bernoulli,complement,categorical}_nb_test.rs` — oracle harness (mbsgd_classifier_test.rs template)
- [ ] `crates/mlrs-py/src/estimators/naive_bayes.rs` + `crates/mlrs-py/src/lib.rs` registration (5 `add_class`) + extend `pyclass_smoke_test.rs`
- [ ] Framework install: none — `cargo test` is the framework; oracle regen needs `/tmp` venv only at fixture-gen time.

## Project Constraints (from CLAUDE.md / AGENTS.md)

- **CubeCL kernels generic over `<F: Float + CubeElement + Pod>` and over runtime** — but NB writes NO new kernel (reductions-only). Any device call uses the existing generic prims.
- **cpu(f64) + rocm(f32) are the correctness gates**; f64-on-rocm skips-with-log; sklearn ≤1e-5 the parity oracle (exact labels the hard gate). `[CLAUDE.md / REQUIREMENTS.md]`
- **Tests strictly separated from source** — `crates/mlrs-algos/tests/<variant>_nb_test.rs`, NEVER an in-source `#[cfg(test)] mod tests` (AGENTS.md §2).
- **CubeCL build-error protocol** — if any cubecl build/launch error arises, read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` before any fix (AGENTS.md §4). Read `…/Cubecl/INDEX.md` before writing any kernel (not expected this phase).
- **thiserror in libs** (`AlgoError`/`BuildError`), `anyhow` only at the PyO3 boundary (project memory `error-handling-convention`).
- **Memory efficiency first-class** — `release_into(pool)` per-segment scratch in the GATHER loop and on re-fit (WR-07); PoolStats no-leak gate per estimator (ROADMAP recurring gate).
- **GSD workflow enforcement** — file edits go through a GSD command (CLAUDE.md).
- **`gsd-tools query` commit/state verbs are no-ops in this build** — verify with git, do the work manually (project memory `gsd-tools-query-verbs-noop`).

## Security Domain

> security_enforcement enabled, ASVS Level 1 (`.planning/config.json`). NB has no auth/session/network surface; the relevant category is input validation at the host→estimator and Python→estimator boundaries (untrusted hyperparameters + data geometry).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | `build()` validates data-independent knobs → `BuildError` (alpha≥0, var_smoothing≥0, finite priors, force_alpha clip); `fit()` validates data-dependent (prior length==n_classes, categorical input non-negative-integer, n_features agreement, geometry) → `AlgoError` BEFORE any device launch. PyO3 maps both → `ValueError` (`build_err_to_py`/`algo_err_to_py`). The validate-before-launch contract (mbsgd/KDE precedent). |
| V6 Cryptography | no | — (no RNG in NB; PRIM-06 RNG not used) |

### Known Threat Patterns for {Rust estimator + PyO3 + CubeCL}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Untrusted hyperparameter (negative alpha/var_smoothing, non-finite prior) reaches a device kernel and drives `log`/`exp` to NaN or an OOB read | Tampering / DoS | `build()` data-independent validation → typed `BuildError` before any data/launch (D-05) |
| Untrusted data geometry (mismatched `rows×cols`, prior length, n_features) causes OOB device read | Tampering | `fit()`/predict geometry guards → `AlgoError::Prim(ShapeMismatch/DimMismatch)` before launch (mbsgd precedent) |
| CategoricalNB negative / non-integer / out-of-range category index → OOB ragged-table lookup | Tampering | `fit()` validates non-negative-integer input; predict guards the lookup index against `n_categories_j` (Pitfall 7) |
| `n` exceeds `u32` launch-grid on a reduce/gemm call | DoS (silent wrong result) | the prims already `u32::try_from(...).expect(...)` the grid (reduce.rs/KDE precedent) |
| PyO3 panic crossing the FFI boundary | DoS | `?` on typed errors, never panic across the boundary; `py.detach` body returns `PyResult` (mbsgd precedent) |

## Sources

### Primary (HIGH confidence — in-repo VERIFIED)
- `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` — builder/`Default`/`build()->Result<_,BuildError>`, classes_ remap, PredictLabels/PredictProba, gemm matvec, release_into
- `crates/mlrs-algos/src/traits.rs` — Fit/PredictLabels/PredictProba/ScoreSamples surface to extend
- `crates/mlrs-algos/src/error.rs` — AlgoError/BuildError two-tier contract
- `crates/mlrs-algos/src/density/kernel_density.rs` — KdKernel/BandwidthSpec enum + host f64 log-sum-exp (single terminal log) precedent
- `crates/mlrs-backend/src/prims/reduce.rs` — row/column_reduce host-segmented (the GATHER substrate), ScalarOp
- `crates/mlrs-backend/src/capability.rs` — skip_f64_with_log, log_oracle_dtype, active_backend_name
- `crates/mlrs-py/src/dispatch.rs` (any_estimator!) + `crates/mlrs-py/src/estimators/linear.rs` (PyMBSGDClassifier: py.detach GIL release, build_err_to_py)
- `crates/mlrs-algos/tests/mbsgd_classifier_test.rs` + `scripts/gen_oracle.py` — oracle harness + fixture-gen convention
- `.planning/research/FEATURES.md §Family 5` — the PINNED per-variant math contract (CITED throughout)
- `.planning/ROADMAP.md §Phase 11` + `.planning/REQUIREMENTS.md` (NB-01…05, PY-06) + `11-CONTEXT.md` (D-01…D-10)

### Secondary (MEDIUM confidence)
- Project memory: `cubecl-cpu-no-shared-memory`, `rocm-is-runnable-gpu-gate`, `backend-test-suite-slow`, `oracle-fixture-regen-needs-venv`, `error-handling-convention`, `gsd-tools-query-verbs-noop` — cross-cutting constraints honored above.

### Tertiary (LOW confidence)
- none — no WebSearch used; sklearn per-variant math taken from FEATURES.md (which cites sklearn 1.9.0 source), not re-derived.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every component VERIFIED in-repo; zero new dependencies (v2 constraint).
- Architecture (GATHER-over-reduce, GEMM joint-LL, host log-sum-exp): HIGH — directly composes validated prims; the no-new-kernel approach is mandated by ROADMAP and proven by the KDE/mbsgd precedents.
- Per-variant math: HIGH (CITED from FEATURES.md, the pinned contract) — but flagged parity risks (Pitfalls 3–7) are where MEDIUM-confidence A3/A4 assumptions live (unseen-category handling, f32-rocm band width).
- Pitfalls: HIGH — cpu-MLIR/scatter-add/log-sum-exp hazards confirmed by project memory + ROADMAP recurring gates.

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (stable — in-repo APIs and a pinned sklearn oracle; refresh if `reduce`/`gemm`/`traits` signatures change or the sklearn oracle version bumps)
