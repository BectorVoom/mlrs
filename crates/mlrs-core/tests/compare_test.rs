//! Tests for the float-comparison core and tolerance policy (Task 1, D-08/D-09).
//!
//! Per AGENTS.md these live in `tests/`, never as a `#[cfg(test)] mod tests`
//! inside `src/`.

use mlrs_core::compare::{assert_slice_close, is_close, NEAR_ZERO_FLOOR};
use mlrs_core::error::BridgeError;
use mlrs_core::tolerance::{Tolerance, F32_TOL, F64_TOL};
use mlrs_core::{assert_close, BridgeError as ReexportedBridgeError};

// --- D-09 behaviors from the plan -------------------------------------------

#[test]
fn within_abs_and_rel_passes() {
    // 1.0 + 9e-6 vs 1.0: abs_err = 9e-6 <= 1e-5 and rel_err = 9e-6 <= 1e-5.
    assert!(is_close(1.0 + 9e-6, 1.0, &F32_TOL));
}

#[test]
fn abs_error_exceeding_floor_fails() {
    // 1.0 + 2e-5 vs 1.0: abs_err = 2e-5 > 1e-5 -> must fail.
    assert!(!is_close(1.0 + 2e-5, 1.0, &F32_TOL));
}

#[test]
fn near_zero_guard_falls_back_to_abs_only() {
    // |expected| < NEAR_ZERO_FLOOR -> abs-only; rel term must not explode.
    assert!(is_close(1e-12, 0.0, &F32_TOL));
    // Still bounded by abs: a value beyond tol.abs from zero fails.
    assert!(!is_close(2e-5, 0.0, &F32_TOL));
}

#[test]
fn both_abs_and_rel_must_pass() {
    // Genuinely different (2.0 vs 1.0): fails both.
    assert!(!is_close(2.0, 1.0, &F32_TOL));

    // Passes rel but fails abs: large magnitude, small relative error,
    // but absolute error exceeds tol.abs. expected=1e6, got=1e6+5 ->
    // abs_err=5 (> 1e-5) BUT rel_err=5e-6 (< 1e-5). BOTH required => fail.
    assert!(!is_close(1_000_000.0 + 5.0, 1_000_000.0, &F32_TOL));

    // Passes abs but fails rel: small magnitude (above the floor), abs error
    // tiny but relative error large. expected=1e-6, got=1e-6+5e-7 ->
    // abs_err=5e-7 (< 1e-5) BUT rel_err=0.5 (> 1e-5). BOTH required => fail.
    assert!(!is_close(1e-6 + 5e-7, 1e-6, &F32_TOL));
}

// --- assert_close informative panic -----------------------------------------

#[test]
fn assert_close_accepts_close_values() {
    assert_close(1.0 + 9e-6, 1.0, &F32_TOL);
    assert_close(-3.5, -3.5, &F64_TOL);
}

#[test]
#[should_panic(expected = "assert_close failed")]
fn assert_close_panics_on_mismatch() {
    assert_close(2.0, 1.0, &F32_TOL);
}

#[test]
fn assert_close_panic_message_reports_errors() {
    let result = std::panic::catch_unwind(|| assert_close(2.0, 1.0, &F32_TOL));
    let err = result.expect_err("expected panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("panic payload is a string");
    assert!(msg.contains("got="), "message should report got: {msg}");
    assert!(msg.contains("expected="), "message should report expected: {msg}");
    assert!(msg.contains("abs_err="), "message should report abs_err: {msg}");
    assert!(msg.contains("rel_err="), "message should report rel_err: {msg}");
}

// --- slice variant ----------------------------------------------------------

#[test]
fn assert_slice_close_passes_for_close_slices() {
    let got = [1.0, 2.0, 3.0];
    let expected = [1.0 + 9e-6, 2.0 - 9e-6, 3.0];
    assert_slice_close(&got, &expected, &F32_TOL);
}

#[test]
#[should_panic(expected = "index 1")]
fn assert_slice_close_reports_offending_index() {
    let got = [1.0, 5.0, 3.0];
    let expected = [1.0, 2.0, 3.0];
    assert_slice_close(&got, &expected, &F32_TOL);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn assert_slice_close_rejects_length_mismatch() {
    assert_slice_close(&[1.0, 2.0], &[1.0], &F32_TOL);
}

// --- tolerance policy structure (D-08) --------------------------------------

#[test]
fn global_defaults_are_1e_minus_5() {
    assert_eq!(F32_TOL, Tolerance::new(1e-5, 1e-5));
    assert_eq!(F64_TOL, Tolerance::new(1e-5, 1e-5));
}

#[test]
fn for_family_returns_global_default_today() {
    // Growth point: all families resolve to the global default in Phase 1.
    assert_eq!(Tolerance::for_family("pca"), F32_TOL);
    assert_eq!(Tolerance::for_family("kmeans"), F32_TOL);
    assert_eq!(Tolerance::for_family("anything"), F32_TOL);
}

#[test]
fn near_zero_floor_is_below_abs_tolerance() {
    // The guard never loosens the abs check: floor < abs tolerance.
    assert!(NEAR_ZERO_FLOOR < F32_TOL.abs);
}

// --- BridgeError typed enum (D-07) ------------------------------------------

#[test]
fn bridge_error_has_distinct_variants_with_messages() {
    let offset = BridgeError::Offset { offset: 3 };
    let nulls = BridgeError::HasNulls { null_count: 2 };
    let misaligned = BridgeError::Misaligned {
        reason: "size not divisible".to_string(),
    };
    let mismatch = BridgeError::DataTypeMismatch {
        expected: "f64".to_string(),
        found: "f32".to_string(),
    };

    assert!(format!("{offset}").contains("offset 3"));
    assert!(format!("{nulls}").contains("2 null"));
    assert!(format!("{misaligned}").contains("size not divisible"));
    assert!(format!("{mismatch}").contains("expected f64"));

    // The re-export points at the same type.
    let _: ReexportedBridgeError = offset;
}
