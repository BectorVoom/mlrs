//! `naive_bayes` — the five sklearn Naive Bayes classifiers (NB-01..05, Phase 11).
//!
//! `GaussianNB` / `MultinomialNB` / `BernoulliNB` / `ComplementNB` /
//! `CategoricalNB` — the v2.0 reductions-only closing bookend: wide-but-shallow,
//! **no new primitive** (only the validated v1 `reduce` prim plus host
//! log/exp/log-sum-exp), and **five mutually-independent, parallel-buildable
//! estimators**.
//!
//! ## D-03 — free functions, NO shared base struct
//!
//! The shared NB math lives as **free functions** in [`nb_common`] (the per-row
//! `log_sum_exp_normalize` for `predict_proba`/`predict_log_proba`, the empirical
//! class log-prior, the argmax/argmin label decode, `accuracy_score`, and the
//! one-owner-per-`(class, feature)` GATHER helper `class_grouped_sum`). The five
//! estimators are **fully independent structs** that *call* these helpers — there
//! is deliberately **NO shared `NbBase` struct** and no inheritance-style
//! coupling (contrast Phase-10's shared `SgdConfig`, which was justified by a
//! single shared prim contract; NB has no shared prim, so the coupling is at the
//! *function* level only). This honors the ROADMAP's "five
//! mutually-independent, parallel-buildable" framing — each estimator is
//! buildable / testable in isolation — while keeping the common math DRY (no 5×
//! duplication).
//!
//! ## Construction — the builder standard (D-01/D-02, inherited from Phase 10)
//!
//! Every estimator is constructed via `Estimator::builder().setter(..).build()?`
//! with sklearn-default field initializers (a bare `builder().build()` reproduces
//! the sklearn default estimator). Data-INDEPENDENT hyperparameters validate at
//! `build() -> Result<_, BuildError>`; data-DEPENDENT checks stay at
//! `fit() -> AlgoError` (D-05 split validation). Python-facing names mirror
//! sklearn per estimator (D-09): `GaussianNB::builder().priors(..).var_smoothing(..)`
//! (NO `alpha`); the four discrete variants use `.class_prior(..).alpha(..)
//! .force_alpha(..).fit_prior(..)` plus per-variant knobs (`binarize`, `norm`,
//! `min_categories`).
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2 — never an in-source
//! `#[cfg(test)] mod tests`).

pub mod nb_common;

pub mod bernoulli_nb;
pub mod categorical_nb;
pub mod complement_nb;
pub mod gaussian_nb;
pub mod multinomial_nb;

pub use bernoulli_nb::BernoulliNB;
pub use categorical_nb::{CategoricalNB, MinCategories};
pub use complement_nb::ComplementNB;
pub use gaussian_nb::GaussianNB;
pub use multinomial_nb::MultinomialNB;
