//! Plan 05-05 — coordinate-descent primitive Wave-0 oracle SCAFFOLD.
//!
//! Nyquist Wave-0 stub: every test referencing the not-yet-existing
//! `prims::coordinate_descent` symbol is `#[ignore]`d and asserts ONLY that the
//! committed `lasso_{f32,f64}_seed42.npz` AND `elastic_net_{f32,f64}_seed42.npz`
//! fixtures load and are shape-well-formed — so this crate COMPILES today against
//! the empty `prims::coordinate_descent` stub. The shared CD kernel serves both
//! Lasso (`l1_ratio=1`) and ElasticNet (D-03), so this prim oracle exercises
//! BOTH fixtures. Plan 05-05 removes `#[ignore]` and wires the real soft-threshold
//! + residual-update oracle (un-normalized form, `l1_reg = α·l1_ratio·n`,
//! duality-gap stop, Pitfall 1).
//!
//! f64 stubs carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips, D-07).
//! Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase};

/// CD fixture geometry (gen_oracle.py CD_N_SAMPLES × CD_N_FEATURES).
const CD_N_SAMPLES: usize = 50;
const CD_N_FEATURES: usize = 8;

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

/// LOAD-NOT-JUST-PRESENT: BOTH the `lasso` and `elastic_net` fixtures load with
/// well-formed X/y/coef/intercept arrays (the shared CD kernel serves both,
/// D-03). WAVE-0 STUB — 05-05 wires the real CD oracle on
/// `prims::coordinate_descent`.
#[test]
#[ignore = "Wave-0 scaffold: prims::coordinate_descent not implemented until plan 05-05"]
fn fixture_loads() {
    let lasso = load_npz(fixture("lasso_f64_seed42.npz")).expect("load lasso_f64");
    assert_len(&lasso, "X", CD_N_SAMPLES * CD_N_FEATURES);
    assert_len(&lasso, "y", CD_N_SAMPLES);
    assert_len(&lasso, "coef", CD_N_FEATURES);
    assert_len(&lasso, "intercept", 1);

    let en = load_npz(fixture("elastic_net_f64_seed42.npz")).expect("load elastic_net_f64");
    assert_len(&en, "X", CD_N_SAMPLES * CD_N_FEATURES);
    assert_len(&en, "coef", CD_N_FEATURES);
    assert_len(&en, "l1_ratio", 1);
}

/// CD soft-threshold reproduces the Lasso sparse `coef_` (exact zeros), f32.
/// WAVE-0 STUB — 05-05 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::coordinate_descent not implemented until plan 05-05"]
fn cd_lasso_coef_match_sklearn_f32() {
    let case = load_npz(fixture("lasso_f32_seed42.npz")).expect("load lasso_f32");
    assert_len(&case, "coef", CD_N_FEATURES);
}

/// CD residual update reproduces the ElasticNet `coef_`, f64 (cpu runs; rocm
/// skips). WAVE-0 STUB — 05-05 wires the real assertion.
#[test]
#[ignore = "Wave-0 scaffold: prims::coordinate_descent not implemented until plan 05-05"]
fn cd_elastic_net_coef_match_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let case = load_npz(fixture("elastic_net_f64_seed42.npz")).expect("load elastic_net_f64");
    assert_len(&case, "coef", CD_N_FEATURES);
}
