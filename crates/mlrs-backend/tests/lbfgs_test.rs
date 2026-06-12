//! Plan 05-06 — L-BFGS primitive Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test referencing the not-yet-existing
//! `prims::lbfgs` symbol is `#[ignore]`d and asserts ONLY that the committed
//! `logistic_binary_{f32,f64}` AND `logistic_multi_{f32,f64}` fixtures load and
//! are shape-well-formed — so this crate COMPILES today against the empty
//! `prims::lbfgs` stub. Plan 05-06 removes `#[ignore]` and wires the real
//! oracle: FIRST the standalone convex-quadratic minimizer invariant
//! (`½xᵀAx − bᵀx` → `x* = A⁻¹b` within 1e-5, RESEARCH Pitfall 5 — isolates "is my
//! L-BFGS correct" from sklearn's path), THEN the stable-softmax loss/grad check.
//! HIGHEST project risk — validate standalone before LogReg consumes it.
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// LogReg fixture geometry (gen_oracle.py LOG_N_FEATURES; n_classes 2 / 3).
const LOG_N_FEATURES: usize = 4;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn assert_len(case: &OracleCase, name: &str, len: usize) {
    let got = case.expect_f64(name).len();
    assert_eq!(
        got, len,
        "fixture array '{name}' should have {len} elements, got {got}"
    );
}

/// LOAD-NOT-JUST-PRESENT: BOTH the `logistic_binary` and `logistic_multi`
/// fixtures load with well-formed coef/intercept/predict_proba arrays (binary
/// coef is 1×n_features, multi is 3×n_features — symmetric over-parameterized
/// softmax). WAVE-0 STUB — 05-06 wires the convex-quadratic + softmax oracle on
/// `prims::lbfgs`.
#[test]
#[ignore = "Wave-0 scaffold: prims::lbfgs not implemented until plan 05-06"]
fn fixture_loads() {
    let bin = load_npz(fixture("logistic_binary_f64_seed42.npz")).expect("load logistic_binary_f64");
    // Binary: sklearn stores a single weight row (1 × n_features).
    assert_len(&bin, "coef", LOG_N_FEATURES);
    assert_len(&bin, "intercept", 1);

    let multi = load_npz(fixture("logistic_multi_f64_seed42.npz")).expect("load logistic_multi_f64");
    // Multiclass: 3 class weight rows (3 × n_features) + 3 intercepts.
    assert_len(&multi, "coef", 3 * LOG_N_FEATURES);
    assert_len(&multi, "intercept", 3);
}

/// Standalone convex-quadratic minimizer invariant `x* = A⁻¹b` within 1e-5, f32.
/// WAVE-0 STUB — 05-06 wires the real invariant (RESEARCH Pitfall 5).
#[test]
#[ignore = "Wave-0 scaffold: prims::lbfgs not implemented until plan 05-06"]
fn lbfgs_convex_quadratic_minimizer_f32() {
    // 05-06 minimizes ½xᵀAx − bᵀx for an SPD A and asserts x ≈ A⁻¹b. Stub asserts
    // a fixture load only for now.
    let case = load_npz(fixture("logistic_binary_f32_seed42.npz")).expect("load logistic_binary_f32");
    assert_len(&case, "coef", LOG_N_FEATURES);
}

/// Stable softmax loss/grad reproduces the reference, f64 (cpu runs; rocm skips).
/// WAVE-0 STUB — 05-06 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::lbfgs not implemented until plan 05-06"]
fn lbfgs_softmax_loss_grad_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("logistic_multi_f64_seed42.npz")).expect("load logistic_multi_f64");
    assert_len(&case, "coef", 3 * LOG_N_FEATURES);
}
