# Phase 10: SGD / Linear-SVM - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 10 delivers four supervised estimators built on the **one genuinely-new
device solver of v2** — the minibatch-SGD prim (`prims/sgd.rs`): `MBSGDClassifier`,
`MBSGDRegressor`, `LinearSVC`, `LinearSVR`. These are the **highest-hyperparameter-count
estimators in the project** (sklearn's SGD surface is ~15–20 knobs vs Ridge's 2),
which is why this discussion focused on the **Rust-native construction API**.

This CONTEXT covers **how the estimators are constructed and configured in Rust**
(builder pattern, enum typing, shared config, validation timing) and how that API
crosses the PyO3 boundary. It does **not** change what the estimators compute —
sklearn parity, the two-pass GATHER kernel, the pinned-deterministic oracle, and
the standalone SGD-prim validation are fixed by the ROADMAP success criteria and
handled by the research spike + plan-phase.

</domain>

<decisions>
## Implementation Decisions

### Construction idiom (builder pattern)
- **D-01:** The **builder pattern is the canonical constructor** for all four
  Phase-10 estimators (e.g. `MBSGDClassifier::builder().loss(Loss::Hinge).alpha(1e-4).build()?`).
  This **replaces the `new(positional)` + `with_opts()` convention** used in
  Phases 4–9 for these estimators. The positional `new()` becomes unworkable at
  ~15+ knobs (cf. the current ceiling: `kernel_ridge::new(kernel, alpha, gamma,
  degree, coef0)`, 5 args).
- **D-02:** The builder is the **going-forward project standard**, not a one-off
  for high-arity estimators. Retrofitting the existing low-arity estimators
  (Ridge, Lasso, ElasticNet, LogisticRegression, etc.) to builders is **deferred
  work, OUT OF SCOPE for Phase 10** (see Deferred Ideas) — Phase 10 only
  *introduces* the standard via the four new estimators.
- **D-03:** The builder **seeds sklearn-exact defaults**. `Estimator::builder().build()`
  (no setters) MUST reproduce scikit-learn's *default* estimator — e.g.
  `MBSGDClassifier`: `loss='hinge'`, `penalty='l2'`, `alpha=1e-4`,
  `learning_rate='optimal'`, `max_iter=1000`, `tol=1e-3`; `MBSGDRegressor`:
  `loss='squared_error'`, `learning_rate='invscaling'`; `LinearSVC`:
  `loss='squared_hinge'`, `dual='auto'`; `LinearSVR`:
  `loss='squared_epsilon_insensitive'`. (The pinned-deterministic oracle overrides
  several of these at test time — `shuffle=False`, fixed `eta0`/schedule, fixed
  `max_iter`, `tol=0` — but the *default* builder must still equal sklearn's default.)

### Categorical knobs: Rust enums, not strings
- **D-04:** Categorical hyperparameters are **Rust enums**, following the existing
  `KernelKind` / `KdKernel` / `BandwidthSpec` precedent (NOT `String` like
  spectral's `affinity`):
  - `Loss::{Hinge, Log, SquaredHinge, SquaredLoss, EpsilonInsensitive, SquaredEpsilonInsensitive}`
  - `Penalty::{L1, L2, ElasticNet}`
  - `LearningRate::{Optimal, InvScaling, Constant, Adaptive}` (incl. Bottou `t0` for `optimal`)
  - Each estimator's builder accepts only the loss variants valid for it.
- **D-05:** Each enum implements **`TryFrom<&str>` using sklearn's spelling**, and
  this is the **single source of truth** for the string↔enum mapping, living in
  `mlrs-algos` (NOT duplicated in the Py wrapper). The PyO3 layer accepts the
  sklearn string, converts via `TryFrom`, and raises a Python **`ValueError`** on
  an unknown/invalid value — matching sklearn's own error behavior.

### Shared config across the four estimators
- **D-06:** **Per-estimator builders**, each exposing **only the knobs valid for
  that estimator** (e.g. `LinearSVC` exposes no `eta0`/`learning_rate`;
  `MBSGDRegressor` exposes `epsilon`). Each `build()` **lowers into one shared
  internal `SgdConfig`/`SgdParams` struct** that the SGD prim consumes — a single
  prim contract, no duplicated prim-parameter plumbing.
- **D-07:** **Solver choice is implicit** (an internal detail, not a user-facing
  builder knob): `MBSGDClassifier`/`MBSGDRegressor` always use the new SGD prim;
  `LinearSVC`/`LinearSVR` resolve `dual='auto'` internally and may reuse the v1
  coordinate-descent (CD) solver for the converged optimum. This mirrors sklearn,
  which does not let the user directly pick the solver for these estimators.

### Validation timing
- **D-08:** **Split validation.** `build() -> Result<Estimator, BuildError>`
  validates **data-independent** hyperparameters at the earliest point
  (`alpha >= 0`, `l1_ratio ∈ [0,1]`, `eta0 > 0`, `epsilon >= 0`, valid
  enum/loss combos). **Data-dependent** checks (shape/geometry, `n_features`
  agreement, label geometry) stay at **`fit() -> AlgoError`** — they cannot be
  known before data arrives. This *diverges* from the Phases 4–9 convention of
  validating everything at `fit`, but is enabled and justified by the builder.
- **D-09:** PyO3 surfaces **`build()` errors (and enum `TryFrom` failures) as
  Python `ValueError` at estimator construction time** — sklearn-faithful. Map
  `BuildError -> ValueError` in the Py wrapper alongside the existing
  `algo_err_to_py` mapping (which continues to handle `fit`-time `AlgoError`).

### Carried-forward conventions (unchanged)
- `fit` still returns `&mut self` (sklearn chaining convention, `traits::Fit`).
- Fitted state stays device-resident (D-03 cross-cutting); host materialization
  only at accessors / oracle boundary.
- Estimators implement the existing trait surface (`Fit` + `Predict` /
  `PredictLabels` / `PredictProba`); `MBSGD*` may also implement `PartialFit`
  (the trait already reserves the supervised `y` slot for Phase-10 MBSGD reuse —
  see `traits.rs` `PartialFit` doc).

### Claude's Discretion
- Exact builder method names, the `BuildError` variant set, and the precise
  `SgdConfig` field layout are left to the planner/researcher, provided they honor
  D-01…D-09.
- Whether `LinearSVC`/`LinearSVR` builders physically reuse the existing CD
  estimator type or wrap it is an implementation detail for the planner (D-07
  only fixes that the choice is internal).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 10: SGD / Linear-SVM" — goal, success criteria
  (standalone SGD-prim validation, pinned-deterministic oracle, two-pass GATHER
  kernel, cpu-launch gate), recurring gates, and the `[v2-P4]` research-spike flag.
- `.planning/REQUIREMENTS.md` — PRIM-10, SGDSVM-01, SGDSVM-02, SGDSVM-03, SGDSVM-04.

### API/construction precedents to follow or supersede
- `crates/mlrs-algos/src/traits.rs` — the `Fit`/`PartialFit`/`Predict`/
  `PredictLabels`/`PredictProba` trait contract the builders' output must satisfy;
  note the `PartialFit` doc explicitly reserves the `y` slot for Phase-10 MBSGD.
- `crates/mlrs-algos/src/linear/logistic.rs` — current `new()` + `with_opts()`
  escape-hatch pattern that the builder supersedes (D-01).
- `crates/mlrs-algos/src/linear/elastic_net.rs` — closest existing precedent for
  `alpha` + `l1_ratio` + `penalty` semantics (`new(alpha, l1_ratio, fit_intercept)`).
- `crates/mlrs-algos/src/linear/ridge.rs` — fit-time validation precedent
  (`InvalidAlpha`) that D-08 partially supersedes for data-independent params.
- `crates/mlrs-algos/src/kernel_ridge/kernel_ridge.rs` §`enum KernelKind` — the
  enum-for-categorical-knob precedent D-04 follows.
- `crates/mlrs-algos/src/cluster/spectral_clustering.rs` — the `affinity: String`
  anti-pattern D-04 deliberately rejects.

### PyO3 boundary
- `crates/mlrs-py/src/estimators/linear.rs` — the `any_estimator! { unfit: { … } }`
  macro that maps Rust constructor fields → the sklearn `#[pyclass]` surface;
  builders + `BuildError`→`ValueError` (D-09) and enum `TryFrom`/`ValueError`
  (D-05) must integrate here. The planner should check whether `any_estimator!`
  needs extension to express a builder-fronted `unfit{}` block.
- `crates/mlrs-py/src/errors.rs` — existing `algo_err_to_py`; extend with the
  `BuildError -> ValueError` mapping (D-09).

### Memory-recall caveat
Recalled memory items (e.g. cubecl-cpu SharedMemory limits, the GATHER idiom)
inform the *prim/kernel* work, not these API decisions — verify against current
source before relying on them in the plan.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mlrs-algos::traits` (`Fit`/`PartialFit`/`Predict`/`PredictLabels`/`PredictProba`):
  the builders only change *construction*; the post-`build()` estimator implements
  this existing surface unchanged.
- Enum precedent: `KernelKind`, `KdKernel`, `BandwidthSpec`, `NComponents` show the
  established way to type categorical params (D-04) — copy the shape, add `TryFrom<&str>`.
- ElasticNet's `alpha`/`l1_ratio`/penalty handling is the nearest analog for the
  SGD penalty math.
- v1 coordinate-descent (`crates/mlrs-algos/src/linear/coordinate_descent.rs`) is
  the candidate solver `LinearSVC`/`LinearSVR` reuse internally (D-07).

### Established Patterns
- All current estimators: `new(positional)` + validate-at-`fit`. Phase 10 breaks
  BOTH for the four estimators — builder construction (D-01) + split validation
  (D-08). This is intentional and scoped to Phase 10's new estimators only.
- PyO3 wrappers are generated through the `any_estimator!` macro whose `unfit{}`
  block enumerates constructor fields — the builder design must remain expressible
  through (a possibly extended) version of this macro.

### Integration Points
- `build()`/enum-conversion errors → `ValueError` via the Py wrapper; `fit`-time
  `AlgoError` continues through `algo_err_to_py`.
- All four estimators' builders → one shared `SgdConfig` → the new `prims/sgd.rs`
  GATHER kernel (the prim contract is the integration seam, validated standalone
  per the ROADMAP before any estimator is wired).

</code_context>

<specifics>
## Specific Ideas

- User's framing verbatim: "design Rust native like (builder pattern)." The intent
  is idiomatic Rust ergonomics for high-arity estimators, while preserving exact
  sklearn parity at the values level (D-03 sklearn-exact defaults).
- `builder().build()` with zero setters == scikit-learn's default estimator is the
  litmus test for D-03.

</specifics>

<deferred>
## Deferred Ideas

- **Retrofit existing estimators to the builder pattern** (Ridge, Lasso,
  ElasticNet, LinearRegression, LogisticRegression, KMeans, PCA, spectral family,
  etc.). The builder is the going-forward standard (D-02), but converting Phases
  4–9 estimators is its own follow-up effort — NOT Phase 10. Capture for a future
  cleanup/consistency phase.
- **Explicit user-facing solver selection** (force SGD vs CD on LinearSVC/SVR).
  Rejected for Phase 10 to stay sklearn-faithful (D-07); revisit only if a
  Rust-native power-user API is later desired.
- **Typestate compile-time validation** of invalid knob combinations. Considered
  under Validation timing; deferred in favor of the lighter `build() -> Result`
  split (D-08) to limit divergence from the rest of the codebase.

</deferred>

---

*Phase: 10-sgd-linear-svm*
*Context gathered: 2026-06-21*
