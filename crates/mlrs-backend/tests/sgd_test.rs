//! Plan 10-01 Wave-0 — sgd_solve (PRIM-10) standalone-validation Nyquist
//! `#[ignore]` scaffolds.
//!
//! These compile today against the Wave-0 stub (`sgd_solve` geometry guard real,
//! compute `todo!()`; the two `sgd_*` kernels SharedMemory-free). The Wave-1 plan
//! un-ignores them and wires the real solve:
//!
//!   - `sgd_convex_objective` — the PRIM-10 standalone convex-problem gate
//!     (RESEARCH §Validation Criterion 1): the prim must minimize a host-reference
//!     convex objective BEFORE any estimator wires it (primitive-first).
//!   - `sgd_cpu_launch` — the cpu-LAUNCH success criterion (Pitfall 1): the two
//!     `sgd_*` kernels must LAUNCH on cpu(MLIR), not merely compile. This is a
//!     Wave-1 SUCCESS CRITERION, not a compile-only assert.
//!
//! The f64 path carries the `skip_f64_with_log` gate (cpu runs f64; rocm
//! skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-backend/tests/`, never an in-source `#[cfg(test)] mod tests`.

use mlrs_backend::capability;

/// PRIM-10 standalone convex-objective gate (RESEARCH §Validation Criterion 1).
/// `#[ignore]` Wave-0: the compute body is `todo!()`; Wave-1 drives the epoch
/// loop and asserts the solved `(coef, intercept)` minimizes the host-reference
/// convex objective within tolerance (f64 strict, f32 documented band).
#[test]
#[ignore = "Wave-1 (plan 10-02) fills sgd_solve compute + the convex-objective gate"]
fn sgd_convex_objective() {
    // skip_f64_with_log: the f64 reference runs on cpu and skips-with-log on rocm.
    let _ = capability::skip_f64_with_log();
    // Wave-1 builds a small strongly-convex squared-error problem, runs sgd_solve
    // to a pinned iterate, and compares against the closed-form host minimum.
}

/// PRIM-10 cpu-LAUNCH gate (Pitfall 1 — compile and launch are different gates).
/// `#[ignore]` Wave-0: the kernels are SharedMemory-free by construction; Wave-1
/// LAUNCHES `sgd_margin` + `sgd_weight_update` on cpu(MLIR) and asserts no
/// `failed to run pass` panic (the 05-02 failure mode).
#[test]
#[ignore = "Wave-1 (plan 10-02) launches sgd_margin + sgd_weight_update on cpu"]
fn sgd_cpu_launch() {
    // Wave-1 launches both kernels with a tiny minibatch and asserts the device
    // round-trip matches a host dot/axpy reference (the cpu-MLIR-safe profile).
}
