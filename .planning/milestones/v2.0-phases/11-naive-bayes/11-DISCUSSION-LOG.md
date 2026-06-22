# Phase 11: Naive Bayes - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 11-naive-bayes
**Areas discussed:** Shared core structure, Knob typing, Validation timing, Trait/method surface, Python-facing naming

**User steer:** "design Rust native, like the builder pattern." The builder
mechanism is already the locked project standard (Phase-10 D-01/D-02), so the
discussion refined how that standard applies to NB's specific shape.

---

## Shared core structure

| Option | Description | Selected |
|--------|-------------|----------|
| Shared NbBase + trait helpers | Shared internal NbBase struct (priors/classes/log-sum-exp/predict_proba/score), builders lower into it. DRY but couples the "independent" five. | |
| Fully independent, duplicate plumbing | Five standalone structs each owning their prior/log-sum-exp/score code. Max parallel-buildability; ~5× duplication. | |
| Shared free functions, no base struct | Common math as free fns in nb_common; estimators independent but call shared helpers. DRY math, no struct coupling. | ✓ |

**User's choice:** Shared free functions, no base struct.
**Notes:** Honors ROADMAP "mutually-independent, parallel-buildable" while keeping common math DRY. Distinct from Phase-10's shared `SgdConfig` struct (justified there by one shared prim contract; NB has no shared prim).

---

## Knob typing

| Option | Description | Selected |
|--------|-------------|----------|
| Option<T> + enums per BandwidthSpec precedent | binarize: Option<f64> (None disables), priors/class_prior: Option<Vec<F>>, min_categories: enum {Infer, Uniform, PerFeature}. Most type-safe. | ✓ |
| Option<T> everywhere, no new enums | Option for None-able knobs; min_categories as Option<Vec<usize>>. Fewer types, loses scalar-vs-per-feature at type level. | |
| You decide (planner's discretion) | Lock Option-for-None principle; leave exact enum-vs-Option per knob to planner. | |

**User's choice:** Option<T> + enums per BandwidthSpec precedent.
**Notes:** Follows existing KdKernel/BandwidthSpec enum precedent; `MinCategories` enum captures sklearn's scalar/array/None polymorphism at the type level.

---

## Validation timing

| Option | Description | Selected |
|--------|-------------|----------|
| Split per D-08 + clip at build() | build(): alpha>=0, var_smoothing>=0, min_categories, force_alpha/alpha clip-with-log. fit(): class_prior length, integer input, n_features. Clip is data-independent. | ✓ |
| Split per D-08, clip at fit() | Same split, but force_alpha clip/warn at fit() — keeps build() purely structural, matches sklearn doing it inside fit. | |
| You decide (planner's discretion) | Lock D-08 split; leave build-vs-fit placement of the clip to planner. | |

**User's choice:** Split per D-08 + clip at build().
**Notes:** force_alpha clip-to-1e-10 + warning treated as a data-independent build-time concern; sklearn parity preserved.

---

## Trait/method surface

| Option | Description | Selected |
|--------|-------------|----------|
| Extend trait surface | Add predict_log_proba to shared trait (with PredictProba), score via shared helper, ComplementNB argmin internal to PredictLabels. Uniform PyO3 wrapping. | ✓ |
| Per-estimator methods, no trait change | predict_log_proba/score as inherent methods; don't touch traits. Less uniform for PyO3. | |
| You decide (planner's discretion) | Lock that predict_log_proba + score exist + PyO3-exposed; CNB argmin internal; trait-vs-inherent to planner. | |

**User's choice:** Extend trait surface.
**Notes:** predict_log_proba added to the shared contract (currently PredictProba only); ComplementNB argmin stays internal to its PredictLabels impl.

---

## Python-facing naming

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror sklearn per-estimator | GaussianNB .priors()/.var_smoothing(); other four .class_prior()/.alpha(). Zero translation in PyO3; accepts sklearn's own asymmetry. | ✓ |
| Unify Rust, translate at PyO3 | One consistent Rust name; PyO3 maps Python priors→class_prior for GaussianNB. Cleaner Rust, adds translation layer. | |
| You decide (planner's discretion) | Lock Python names sklearn-exact; leave mirror-vs-translate to planner. | |

**User's choice:** Mirror sklearn per-estimator.
**Notes:** Faithful get_params/set_params; zero name-translation layer in bindings.

---

## Claude's Discretion

- Exact builder method names beyond the sklearn-mirrored hyperparameters, the `BuildError` variant set, and per-estimator struct field layout (honor D-01…D-10).
- Exact factoring of the `nb_common` free-function module (no base/config struct; no 5× duplication).
- Device layout of the GATHER kernel and CategoricalNB ragged `feature_log_prob_` (fixed only by ROADMAP success criteria + FEATURES.md parity notes).

## Deferred Ideas

- NB `partial_fit` — out of scope (PY-06 scopes partial_fit to IncrementalPCA/MBSGD only); candidate for a future streaming-NB milestone.
- Retrofitting Phases 4–9 low-arity estimators to builders — already deferred by Phase-10 D-02.
- A shared `NbBase` struct / trait-object NB abstraction — explicitly rejected this phase (D-03); revisit only if a future variant justifies it.
