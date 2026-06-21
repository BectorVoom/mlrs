# Phase 11: Naive Bayes - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 11 delivers the five sklearn-compatible Naive Bayes classifiers —
`GaussianNB`, `MultinomialNB`, `BernoulliNB`, `ComplementNB`, `CategoricalNB` —
as the v2.0 **reductions-only closing bookend**: wide-but-shallow, **no new
primitive** (only the validated v1 reduce prim plus log/exp + log-sum-exp), and
five mutually-independent, parallel-buildable estimators.

This discussion focused on the **Rust-native construction API** — the user's
explicit steer ("design Rust native, like the builder pattern"). It covers
**how the five estimators are constructed, typed, validated, and exposed in
Rust** and how that surface crosses the PyO3 boundary (PY-06). It does **NOT**
change what they compute: the per-variant likelihood math, smoothing
denominators, the one-owner-per-(class,feature) GATHER kernel, log-sum-exp /
`var_smoothing`, exact-label gate, and sklearn ≤1e-5 parity are FIXED by the
ROADMAP success criteria + `FEATURES.md` and are handled by plan-phase
(research-phase is flagged skippable for this family).

**The builder mechanism itself is already a locked project standard from Phase
10 (D-01/D-02) — it is the going-forward convention, not a new decision here.**
This phase's decisions are about how that standard *applies to NB's specific
shape*: five near-identical classifiers, polymorphic/None-able knobs, a new
`predict_log_proba` method, sklearn's own cross-estimator naming asymmetry, and
ComplementNB's argmin decision rule.

</domain>

<decisions>
## Implementation Decisions

### Construction idiom — builder pattern (carried-forward standard)
- **D-01 (inherited, Phase-10 D-01/D-02):** All five NB estimators are
  constructed via the **builder pattern** — `Estimator::builder().setter(..).build()?`
  — the going-forward project standard. NOT `new(positional)`/`with_opts()`.
- **D-02 (inherited, Phase-10 D-03):** `Estimator::builder().build()` with no
  setters MUST reproduce scikit-learn's **default** estimator. Per-variant
  defaults (from `FEATURES.md`): GaussianNB `var_smoothing=1e-9`, `priors=None`;
  MultinomialNB/BernoulliNB/ComplementNB/CategoricalNB `alpha=1.0`,
  `force_alpha=True`, `fit_prior=True`, `class_prior=None`; BernoulliNB
  `binarize=0.0`; ComplementNB `norm=False`; CategoricalNB `min_categories=None`.

### Shared structure — free functions, no base struct (NEW, this phase)
- **D-03:** Shared NB math lives as **free functions in a `nb_common` module**
  (e.g. log-sum-exp normalize for `predict_proba`/`predict_log_proba`, empirical
  class-prior from `class_count_`, argmax/argmin label decode). The **five
  estimators stay fully independent structs** that *call* these helpers — there
  is **NO shared `NbBase` struct** and no inheritance-style coupling. This
  honors the ROADMAP's "five mutually-independent, parallel-buildable" framing
  (each estimator buildable/testable in isolation) while keeping the common math
  DRY (no 5× duplication). Contrast Phase-10's shared `SgdConfig` struct — that
  was justified by a single shared prim contract; NB has no shared prim, so the
  coupling is at the *function* level only, not a config/state struct.

### Knob typing — Option<T> + enums per BandwidthSpec precedent (NEW)
- **D-04:** None-meaning-default and polymorphic knobs are typed Rust-natively,
  following the existing `KdKernel`/`BandwidthSpec` enum + `Option` precedent
  (NOT sklearn's stringly/None-overloaded scalars):
  - `binarize: Option<f64>` — `None` **disables** binarization (BernoulliNB
    assumes already-binary input); `Some(t)` thresholds `x > t → 1`.
  - `priors` / `class_prior: Option<Vec<F>>` — `None` → empirical priors from
    `class_count_`; `Some(..)` → user-supplied.
  - `min_categories` (CategoricalNB) — a **dedicated enum**
    `MinCategories::{ Infer, Uniform(usize), PerFeature(Vec<usize>) }` capturing
    sklearn's scalar-vs-per-feature-vs-None polymorphism at the type level
    (mirrors the `BandwidthSpec` precedent for a value-shaped knob).
- **D-04a (inherited, Phase-10 D-04/D-05):** Any genuinely-categorical knob would
  be a Rust enum with `TryFrom<&str>` (sklearn spelling) as the single source of
  truth in `mlrs-algos`, PyO3 mapping unknown values → Python `ValueError`. NB
  has few such knobs (most are float/bool/Option); the polymorphic-value enums
  above (`MinCategories`) are the notable case.

### Validation timing — split per D-08, force_alpha clip at build() (NEW)
- **D-05 (inherited split, Phase-10 D-08/D-09):** **Data-independent** knobs
  validate at **`build() -> Result<_, BuildError>`** (earliest point);
  **data-dependent** checks stay at **`fit() -> AlgoError`**. PyO3 surfaces
  `BuildError` (and enum `TryFrom` failures) as Python `ValueError` at
  construction time; `fit`-time `AlgoError` continues via the existing
  `algo_err_to_py` mapping.
  - **build():** `alpha >= 0`, `var_smoothing >= 0`, `min_categories` entries
    non-negative, `class_prior`/`priors` entries valid (finite, non-negative).
  - **fit():** `class_prior`/`priors` **length == n_classes** (data-dependent),
    CategoricalNB non-negative-integer-encoded input, `n_features` agreement.
- **D-06:** The sklearn **`force_alpha` parity nuance** — when `force_alpha=False`
  and `alpha < 1e-10`, sklearn clips `alpha` to `1e-10` **and emits a warning** —
  is handled at **`build()`** (it is a data-independent concern). The clip +
  warning must reproduce sklearn's behavior so parity holds.

### Trait/method surface — extend the shared trait surface (NEW)
- **D-07:** Add **`predict_log_proba`** to the shared estimator trait surface
  (alongside the existing `PredictProba` — see `traits.rs`), since `FEATURES.md`
  lists it for all five and it is not yet in the contract. Keep `PredictLabels`
  for `predict`. Add **`score`** via a shared helper (accuracy for these
  classifiers, mirroring sklearn `ClassifierMixin.score`). This keeps the PyO3
  wrapping uniform across the five estimators.
- **D-08:** **ComplementNB's `argmin` decision rule stays internal** to its
  `PredictLabels` impl — same trait as the other four, different internal
  decision (CNB picks the class whose *complement* fits worst; note the sign).
  Do not special-case it in the trait or the PyO3 layer.

### Python-facing naming — mirror sklearn per-estimator (NEW)
- **D-09:** Builder method names and PY-06 Python-facing hyperparameter names
  **mirror sklearn exactly, per estimator**, accepting sklearn's own
  cross-estimator asymmetry: `GaussianNB::builder().priors(..).var_smoothing(..)`
  (no `alpha`); the other four use `.class_prior(..).alpha(..)`. This gives
  **zero name-translation in the PyO3 layer** and faithful `get_params`/
  `set_params`. No unification + translation layer.

### partial_fit — OUT OF SCOPE for Phase 11
- **D-10:** Although sklearn's NB estimators all support `partial_fit`, **PY-06
  scopes `partial_fit` to IncrementalPCA/MBSGD only** (ROADMAP success criterion
  #4). NB `partial_fit` is **NOT in this phase** — adding it would be scope
  creep. The five NB estimators implement `Fit` (not `PartialFit`).

### Claude's Discretion
- Exact builder method names beyond the sklearn-mirrored hyperparameters, the
  `BuildError` variant set, and the precise field layout of each (independent)
  estimator struct are left to the planner/researcher, provided they honor
  D-01…D-10.
- The exact factoring of the `nb_common` free-function module (which helpers,
  signatures) is the planner's, provided no shared base/config struct is
  introduced (D-03) and the common math is not 5×-duplicated.
- How the one-owner-per-(class,feature) GATHER kernel and the CategoricalNB
  ragged `feature_log_prob_` (list-of-matrices) are laid out on device is an
  implementation detail for plan-phase (fixed only by the ROADMAP success
  criteria + `FEATURES.md` parity notes).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition, requirements & per-variant math
- `.planning/ROADMAP.md` §"Phase 11: Naive Bayes" — goal, the four Success
  Criteria (GATHER kernel + cpu-launch gate, per-variant alpha/denominator,
  predict_proba rows sum to 1 via log-sum-exp, exact labels, PY-06), recurring
  gates (`skip_f64_with_log`, f32-on-rocm band, GATHER idiom Pitfall 2,
  log-sum-exp + var_smoothing Pitfall 9, PoolStats memory gate), and the PY-06
  placement decision (final cross-cutting Python-surface sign-off).
- `.planning/REQUIREMENTS.md` — NB-01, NB-02, NB-03, NB-04, NB-05, PY-06.
- `.planning/research/FEATURES.md` §"Family 5 — Naive Bayes (Phase 11)" — the
  PINNED per-variant likelihood math, defaults, attributes, and ⚠ parity-risk
  notes for each of the five estimators (global-variance `epsilon_`,
  `alpha·n_features` denominators, Bernoulli `(1−x)·log(1−p)` non-occurrence
  term, Complement complement-weights + `norm` + argmin sign, Categorical ragged
  `feature_log_prob_` + `min_categories` padding). This is the math contract.

### API/construction precedents to follow (builder standard from Phase 10)
- `.planning/phases/10-sgd-linear-svm/10-CONTEXT.md` — the builder-pattern
  standard (D-01 builder canonical, D-02 going-forward standard, D-03 sklearn
  defaults, D-04/D-05 enum + `TryFrom<&str>` single-source-of-truth, D-08 split
  validation, D-09 `BuildError`→`ValueError`). Phase 11 inherits all of these.
- `crates/mlrs-algos/src/linear/mbsgd_classifier.rs` — reference builder impl
  (`builder()` → `Builder` with `Default` seeding sklearn defaults →
  `build() -> Result<_, BuildError>`); classifier shape (`classes_` remap +
  `PredictLabels` + `PredictProba`) closest to the NB estimators.
- `crates/mlrs-algos/src/traits.rs` — the `Fit`/`PredictLabels`/`PredictProba`
  trait contract to extend with `predict_log_proba` (D-07); `fit` returns
  `&mut self`, fitted state device-resident.
- `crates/mlrs-algos/src/density/kernel_density.rs` — `KdKernel` / `BandwidthSpec`
  enum precedent for the polymorphic-value knob typing (D-04, `MinCategories`).
- `crates/mlrs-algos/src/error.rs` — `AlgoError` / `BuildError` to extend.

### Cross-cutting conventions
- `AGENTS.md` — tests strictly separated from source (`tests/` or `*_test.rs`,
  never in-source `#[cfg(test)] mod tests`); CubeCL error-guideline protocol.
- `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — project
  conventions and the oracle/gate harness (cpu f64 + rocm f32, sklearn ≤1e-5).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **v1 reduce prim** (`mlrs-backend`): the only primitive NB needs — class-
  conditional sums/counts via the validated reduce path. No new prim (ROADMAP).
- **Builder reference** (`crates/mlrs-algos/src/linear/mbsgd_classifier.rs` and
  the other three Phase-10 estimators): copy the `builder()`/`Default`/`build()`
  shape; NB has fewer/simpler knobs.
- **Enum-typing precedent** (`density/kernel_density.rs` `KdKernel`/`BandwidthSpec`):
  template for `MinCategories` and any `TryFrom<&str>` knob.
- **Classifier plumbing** (`linear/logistic.rs`, `linear/mbsgd_classifier.rs`):
  `classes_` inference/remap, `PredictLabels` + `PredictProba` impls — directly
  analogous to NB.
- **PyO3 machinery** (`mlrs-py`, `any_estimator!`): v2 adds zero binding
  infrastructure; reuse the shipped `#[pyclass]` + dtype-suffixed accessor
  pattern, `algo_err_to_py`, and `BuildError`→`ValueError` mapping.

### Established Patterns
- **Builder is the standard** (Phase-10 D-01/D-02) — already the convention; this
  phase applies it, doesn't introduce it.
- **Split validation** (Phase-10 D-08) — build() data-independent, fit() data-
  dependent.
- **Device-resident fitted state** (cross-cutting D-03) — `theta_`/`var_`/
  `feature_log_prob_`/`class_log_prior_` etc. stay on device; host
  materialization only at accessors / oracle boundary.
- **Generic over `<F: Float + CubeElement + Pod>`** and over runtime — single
  generic codebase, f32/f64 runtime dispatch at the PyO3 layer.

### Integration Points
- `crates/mlrs-algos/src/traits.rs` — add `predict_log_proba` to the trait
  surface (D-07).
- A new `crates/mlrs-algos/src/naive_bayes/` module (mod.rs + five estimator
  files + `nb_common.rs` free-fn helpers per D-03), wired into `lib.rs`.
- `crates/mlrs-py` — register the five `#[pyclass]` estimators; PY-06 final
  cross-cutting Python-surface sign-off (all v2 estimators registered, dtype
  accessors complete, `estimator_checks` re-triaged across the full v2 surface).

</code_context>

<specifics>
## Specific Ideas

- The user's framing: **"design Rust native, like the builder pattern."** The
  intent is that the NB surface should feel idiomatic Rust (builders, enums,
  `Option`, `Result`-returning validation) — already aligned with the locked
  Phase-10 standard, with NB-specific refinements captured in D-03…D-09.
- Keep the five estimators independently buildable/testable (D-03) — the
  "parallel-buildable" property is a deliberately valued trait, not incidental.

</specifics>

<deferred>
## Deferred Ideas

- **NB `partial_fit`** — sklearn supports it for all five, but PY-06 scopes
  `partial_fit` to IncrementalPCA/MBSGD only (D-10). A future milestone could
  add streaming NB (running mean/variance merge for GaussianNB, count
  accumulation for the discrete variants).
- **Retrofitting Phases 4–9 low-arity estimators to builders** — already
  deferred by Phase-10 D-02; out of scope here too.
- **A shared `NbBase` struct / trait-object NB abstraction** — explicitly
  rejected for this phase (D-03 chose free functions). If a sixth NB variant or
  cross-variant tooling ever justifies it, revisit.

</deferred>

---

*Phase: 11-naive-bayes*
*Context gathered: 2026-06-21*
