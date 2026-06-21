//! Plan 10-01 Wave-0 — `sgd_config` enum `TryFrom` + builder `build()` scaffolds
//! (SGDSVM-01..04, D-04/D-05/D-08).
//!
//! The `try_from_*` tests exercise ONLY the Wave-0 enum surface (the
//! single-source `TryFrom<&str>` accepting sklearn spellings + legacy aliases,
//! D-05) so they are LIVE (not `#[ignore]`). The `build_rejects_bad_alpha`
//! scaffold is `#[ignore]` until the Wave-1 plan fills the `build()` validation
//! predicates (D-08 — the SIGNATURE is final now, the predicates land Wave-1).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_algos::error::BuildError;
use mlrs_algos::linear::mbsgd_classifier::MBSGDClassifier;
use mlrs_algos::linear::sgd_config::{LearningRate, Loss, Penalty};

/// D-05: `TryFrom<&str>` accepts the sklearn spellings AND the legacy aliases
/// (`log`/`log_loss`, `squared_error`/`squared_loss`) for every enum.
#[test]
fn try_from_accepts_sklearn_spellings() {
    // Loss — sklearn spellings.
    assert_eq!(Loss::try_from("hinge").unwrap(), Loss::Hinge);
    assert_eq!(Loss::try_from("squared_hinge").unwrap(), Loss::SquaredHinge);
    assert_eq!(
        Loss::try_from("epsilon_insensitive").unwrap(),
        Loss::EpsilonInsensitive
    );
    assert_eq!(
        Loss::try_from("squared_epsilon_insensitive").unwrap(),
        Loss::SquaredEpsilonInsensitive
    );
    // Loss — legacy aliases.
    assert_eq!(Loss::try_from("log").unwrap(), Loss::Log);
    assert_eq!(Loss::try_from("log_loss").unwrap(), Loss::Log);
    assert_eq!(Loss::try_from("squared_error").unwrap(), Loss::SquaredLoss);
    assert_eq!(Loss::try_from("squared_loss").unwrap(), Loss::SquaredLoss);

    // Penalty.
    assert_eq!(Penalty::try_from("l1").unwrap(), Penalty::L1);
    assert_eq!(Penalty::try_from("l2").unwrap(), Penalty::L2);
    assert_eq!(
        Penalty::try_from("elasticnet").unwrap(),
        Penalty::ElasticNet
    );

    // LearningRate.
    assert_eq!(
        LearningRate::try_from("optimal").unwrap(),
        LearningRate::Optimal
    );
    assert_eq!(
        LearningRate::try_from("invscaling").unwrap(),
        LearningRate::InvScaling
    );
    assert_eq!(
        LearningRate::try_from("constant").unwrap(),
        LearningRate::Constant
    );
    assert_eq!(
        LearningRate::try_from("adaptive").unwrap(),
        LearningRate::Adaptive
    );
}

/// D-05/D-08: an unknown enum string maps to the typed `BuildError::Unknown*`
/// (NOT a panic) — the single-mapper-to-ValueError contract (D-09).
#[test]
fn try_from_rejects_unknown() {
    match Loss::try_from("not_a_loss") {
        Err(BuildError::UnknownLoss { value }) => assert_eq!(value, "not_a_loss"),
        other => panic!("expected UnknownLoss, got {other:?}"),
    }
    match Penalty::try_from("nope") {
        Err(BuildError::UnknownPenalty { value }) => assert_eq!(value, "nope"),
        other => panic!("expected UnknownPenalty, got {other:?}"),
    }
    match LearningRate::try_from("warp") {
        Err(BuildError::UnknownLearningRate { value }) => assert_eq!(value, "warp"),
        other => panic!("expected UnknownLearningRate, got {other:?}"),
    }
}

/// The Wave-0 builder lowers the (default-valid) params into a `SgdConfig` and
/// returns `Ok` — the `build()` SIGNATURE is final now (D-01). This is the D-03
/// litmus seed: the default builder encodes sklearn's `SGDClassifier` defaults.
#[test]
fn build_default_lowers_sklearn_defaults() {
    let est = MBSGDClassifier::<f32>::builder()
        .build::<f32>()
        .expect("default builder lowers valid sklearn defaults");
    let cfg = est.config();
    assert_eq!(cfg.loss, Loss::Hinge);
    assert_eq!(cfg.penalty, Penalty::L2);
    assert_eq!(cfg.alpha, 1e-4);
    assert_eq!(cfg.learning_rate, LearningRate::Optimal);
    assert_eq!(cfg.max_iter, 1000);
    assert_eq!(cfg.tol, 1e-3);
    assert_eq!(cfg.eta0, 0.01);
    assert_eq!(cfg.power_t, 0.5);
    assert_eq!(cfg.l1_ratio, 0.15);
}

/// Wave-1 SUCCESS CRITERION (D-08): `build()` rejects a negative `alpha` with
/// `BuildError::InvalidAlpha` BEFORE any data is seen. `#[ignore]` until the
/// Wave-1 plan fills the validation predicates (the Wave-0 stub `build()` does
/// not yet validate — the SIGNATURE is final, the body is a stub).
#[test]
#[ignore = "Wave-1 (plan 10-02) fills build() validation predicates (D-08)"]
fn build_rejects_bad_alpha() {
    match MBSGDClassifier::<f32>::builder().alpha(-1.0).build::<f32>() {
        Err(BuildError::InvalidAlpha { alpha, .. }) => assert_eq!(alpha, -1.0),
        other => panic!("expected InvalidAlpha, got {:?}", other.is_ok()),
    }
}
