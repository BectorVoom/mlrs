//! Plan 12-01 Wave-1 — coexistence/importability smoke test for the NEW
//! `mlrs_algos::typestate` surface (D-03/D-07).
//!
//! This is a COMPILE-LEVEL smoke test only: it proves the new module is
//! importable, that the `Unfit`/`Fitted` markers are zero-sized, and that both
//! satisfy the sealed `State` bound. The structural predict-before-fit PROOF
//! (a compile-fail assertion) is the Plan 03 `trybuild` gate and is deliberately
//! NOT duplicated here; the behavior tests live with the Plan 02 shells.
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_algos::typestate::{Fitted, State, Unfit};

/// Generic helper that compiles only if `S` satisfies the sealed [`State`]
/// bound — invoking it for a type is a static proof of `State` membership.
fn assert_state<S: State>() {}

#[test]
fn markers_are_zero_sized() {
    // Unfit/Fitted are pure type-level tags: they must add zero bytes when
    // carried as a `PhantomData<S>` state slot on an estimator.
    assert_eq!(std::mem::size_of::<Unfit>(), 0);
    assert_eq!(std::mem::size_of::<Fitted>(), 0);
}

#[test]
fn markers_satisfy_sealed_state_bound() {
    // Both markers must satisfy the sealed `State` bound. If either failed to
    // impl `State`, these calls would not compile.
    assert_state::<Unfit>();
    assert_state::<Fitted>();
}

#[test]
fn typestate_module_is_importable() {
    // The markers are constructible from the public path — confirms the module
    // is wired into lib.rs and reachable as `mlrs_algos::typestate::*`.
    let _unfit = Unfit;
    let _fitted = Fitted;
}
