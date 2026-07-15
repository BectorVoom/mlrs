//! PRIM-10 `sgd_solve` standalone validation (Wave-1, plan 10-02).
//!
//! Three families of gate live here:
//!
//!   - `sgd_cpu_launch` / `sgd_margin_matches_host` / `sgd_weight_update_matches_host`
//!     — the cpu-LAUNCH success criterion (Pitfall 1): the two `sgd_*` kernels must
//!     LAUNCH on cpu(MLIR), not merely compile, and their device round-trip must
//!     match a plain host dot/axpy reference (f32 + f64).
//!   - `sgd_convex_objective` — the PRIM-10 standalone convex-problem gate
//!     (RESEARCH §Validation Criterion 1): `sgd_solve` on a strongly-convex
//!     squared-error system must reach the host closed-form OLS optimum within
//!     tolerance (f64 strict 1e-5, f32 documented band) BEFORE any estimator wires
//!     it (primitive-first).
//!   - `dloss_*` / `schedule_*` — the host helper unit tests; a CONSTANT-schedule
//!     case is asserted FIRST to isolate the `optimal` t0 math (A1 / Pitfall 3).
//!
//! The f64 path carries the `skip_f64_with_log` gate (cpu runs f64; rocm
//! skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-backend/tests/`, never an in-source `#[cfg(test)] mod tests`.

use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::sgd::{
    dloss, loss_id, optimal_t0, schedule_eta, sgd_solve, SgdLoss, SgdParams, SgdSchedule,
};
use mlrs_backend::runtime::{self, ActiveRuntime};

use mlrs_kernels::sgd::{sgd_grad, sgd_l1_shrink, sgd_margin, sgd_weight_update};

// ===========================================================================
// Host references (the byte-exact f64 truth the device kernels must match).
// ===========================================================================

/// Host `p[i] = Σ_j x[i*d+j]·w[j] + bias` over the `b × d` minibatch.
fn host_margin(x: &[f64], w: &[f64], bias: f64, b: usize, d: usize) -> Vec<f64> {
    (0..b)
        .map(|i| {
            let mut acc = 0.0f64;
            for j in 0..d {
                acc += x[i * d + j] * w[j];
            }
            acc + bias
        })
        .collect()
}

/// Host `w[j] -= eta·inv_b·Σ_i g[i]·x[i*d+j]` over the `b × d` minibatch.
fn host_weight_update(
    x: &[f64],
    g: &[f64],
    w: &[f64],
    eta: f64,
    inv_b: f64,
    b: usize,
    d: usize,
) -> Vec<f64> {
    (0..d)
        .map(|j| {
            let mut grad = 0.0f64;
            for i in 0..b {
                grad += g[i] * x[i * d + j];
            }
            w[j] - eta * inv_b * grad
        })
        .collect()
}

/// Reinterpret an `f64` as the runtime float `F` (f32 / f64) for host fills.
fn to_f<F: bytemuck::Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("sgd is f32/f64 only"),
    }
}

/// Inverse of [`to_f`]: promote an `F` (f32 / f64) device value to `f64`.
fn from_f<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("sgd is f32/f64 only"),
    }
}

// ===========================================================================
// Kernel launch + host-reference gates (the cpu-LAUNCH success criterion).
// ===========================================================================

/// Launch `sgd_margin` over a `b × d` minibatch and read the `p[]` margin back.
fn launch_margin<F: Float + CubeElement + bytemuck::Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    w: &[f64],
    bias: f64,
    b: usize,
    d: usize,
) -> Vec<f64> {
    let x_f: Vec<F> = x.iter().map(|&v| to_f::<F>(v)).collect();
    let w_f: Vec<F> = w.iter().map(|&v| to_f::<F>(v)).collect();
    let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &x_f);
    let w_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &w_f);
    // The intercept is device-resident (length-1 array) in the reworked kernel.
    let bias_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &[to_f::<F>(bias)]);
    let p_handle = pool.acquire(b * std::mem::size_of::<F>());

    let client = pool.client().clone();
    let block = 256u32;
    let cubes = ((b as u32) + block - 1) / block.max(1);
    let count = CubeCount::Static(cubes.max(1), 1, 1);
    let dim = CubeDim {
        x: block,
        y: 1,
        z: 1,
    };
    let x_arg = unsafe { ArrayArg::from_raw_parts(x_dev.handle().clone(), b * d) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
    let bias_arg = unsafe { ArrayArg::from_raw_parts(bias_dev.handle().clone(), 1) };
    let p_arg = unsafe { ArrayArg::from_raw_parts(p_handle.clone(), b) };
    sgd_margin::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        x_arg,
        w_arg,
        bias_arg,
        p_arg,
        0u32, // row_offset = 0: the uploaded x IS the batch.
        b as u32,
        d as u32,
    );

    let p_dev = DeviceArray::<ActiveRuntime, F>::from_raw(p_handle.clone(), b);
    let host: Vec<f64> = p_dev.to_host(pool).iter().map(|&v| from_f::<F>(v)).collect();
    x_dev.release_into(pool);
    w_dev.release_into(pool);
    bias_dev.release_into(pool);
    pool.release(p_handle, b * std::mem::size_of::<F>());
    host
}

/// Launch `sgd_weight_update` over a `b × d` minibatch and read the updated `w[]`.
fn launch_weight_update<F: Float + CubeElement + bytemuck::Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[f64],
    g: &[f64],
    w: &[f64],
    eta: f64,
    inv_b: f64,
    b: usize,
    d: usize,
) -> Vec<f64> {
    let x_f: Vec<F> = x.iter().map(|&v| to_f::<F>(v)).collect();
    let g_f: Vec<F> = g.iter().map(|&v| to_f::<F>(v)).collect();
    let w_f: Vec<F> = w.iter().map(|&v| to_f::<F>(v)).collect();
    let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &x_f);
    let g_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &g_f);
    let w_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &w_f);

    let client = pool.client().clone();
    let block = 256u32;
    let cubes = ((d as u32) + block - 1) / block.max(1);
    let count = CubeCount::Static(cubes.max(1), 1, 1);
    let dim = CubeDim {
        x: block,
        y: 1,
        z: 1,
    };
    let x_arg = unsafe { ArrayArg::from_raw_parts(x_dev.handle().clone(), b * d) };
    let g_arg = unsafe { ArrayArg::from_raw_parts(g_dev.handle().clone(), b) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) };
    sgd_weight_update::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        x_arg,
        g_arg,
        w_arg,
        to_f::<F>(eta),
        to_f::<F>(inv_b),
        to_f::<F>(1.0), // l2_factor = 1.0: the pure gradient step (host ref has no shrink).
        0u32,           // row_offset = 0: the uploaded x IS the batch.
        d as u32,
        b as u32,
    );

    let host: Vec<f64> = w_dev.to_host(pool).iter().map(|&v| from_f::<F>(v)).collect();
    x_dev.release_into(pool);
    g_dev.release_into(pool);
    w_dev.release_into(pool);
    host
}

/// Deterministic `b × d` minibatch + length-`d` weight fill (no RNG — the gates
/// compare against a host reference, so any reproducible spread suffices).
fn fixture(b: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..b * d)
        .map(|i| ((i % 11) as f64) * 0.13 - 0.6)
        .collect();
    let w: Vec<f64> = (0..d).map(|j| 0.25 * (j as f64) - 0.4).collect();
    (x, w)
}

fn run_margin_match<F: Float + CubeElement + bytemuck::Pod>(label: &str) {
    let (b, d) = (5usize, 4usize);
    let (x, w) = fixture(b, d);
    let bias = 0.37f64;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let dev = launch_margin::<F>(&mut pool, &x, &w, bias, b, d);
    let host = host_margin(&x, &w, bias, b, d);

    // f64 strict 1e-5; f32 a documented round-off band.
    let tol = if std::mem::size_of::<F>() == 8 {
        1e-5
    } else {
        1e-4
    };
    for i in 0..b {
        assert!(
            (dev[i] - host[i]).abs() <= tol,
            "[{label}] sgd_margin p[{i}]={} != host {} (tol {tol})",
            dev[i],
            host[i]
        );
    }
}

fn run_weight_update_match<F: Float + CubeElement + bytemuck::Pod>(label: &str) {
    let (b, d) = (5usize, 4usize);
    let (x, w) = fixture(b, d);
    let g: Vec<f64> = (0..b).map(|i| 0.4 * (i as f64) - 0.9).collect();
    let eta = 0.05f64;
    let inv_b = 1.0 / b as f64;

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let dev = launch_weight_update::<F>(&mut pool, &x, &g, &w, eta, inv_b, b, d);
    let host = host_weight_update(&x, &g, &w, eta, inv_b, b, d);

    let tol = if std::mem::size_of::<F>() == 8 {
        1e-5
    } else {
        1e-4
    };
    for j in 0..d {
        assert!(
            (dev[j] - host[j]).abs() <= tol,
            "[{label}] sgd_weight_update w[{j}]={} != host {} (tol {tol})",
            dev[j],
            host[j]
        );
    }
}

/// PRIM-10 cpu-LAUNCH gate (Pitfall 1 — compile and launch are different gates).
/// LAUNCHES `sgd_margin` + `sgd_weight_update` on the active backend and asserts
/// the device round-trip matches a host dot/axpy reference for f32 AND f64 (the
/// f64 arm runs on cpu, skips-with-log on rocm). No `failed to run pass` panic.
#[test]
fn sgd_cpu_launch() {
    let _ = env_logger::builder().is_test(true).try_init();

    // f32 always runs (portable on every backend).
    run_margin_match::<f32>("f32");
    run_weight_update_match::<f32>("f32");

    // f64 runs on cpu, skips-with-log on rocm (D-07).
    if capability::skip_f64_with_log() {
        return;
    }
    run_margin_match::<f64>("f64");
    run_weight_update_match::<f64>("f64");
}

/// Over-provisioned launch: threads beyond `b` (margin) write nothing. We launch
/// with a deliberately oversized grid and assert the `p[]` slice is exactly the
/// host margin (the bounds-check `if i < b` holds — no out-of-bounds write).
#[test]
fn sgd_margin_matches_host() {
    let _ = env_logger::builder().is_test(true).try_init();
    run_margin_match::<f32>("f32");
    if capability::skip_f64_with_log() {
        return;
    }
    run_margin_match::<f64>("f64");
}

#[test]
fn sgd_weight_update_matches_host() {
    let _ = env_logger::builder().is_test(true).try_init();
    run_weight_update_match::<f32>("f32");
    if capability::skip_f64_with_log() {
        return;
    }
    run_weight_update_match::<f64>("f64");
}

// ===========================================================================
// dloss / schedule host-helper unit tests (constant case isolated FIRST).
// ===========================================================================

/// `dloss` matches the RESEARCH §SGD-Math subgradient table at sample points.
#[test]
fn dloss_table_matches_research() {
    let eps = 0.1f64;

    // Hinge: z = p·y; z<=1 → -y else 0. At p=0.5,y=1 → z=0.5<=1 → -1.
    assert_eq!(dloss(SgdLoss::Hinge, 0.5, 1.0, eps), -1.0);
    // z = 2.0 > 1 → 0.
    assert_eq!(dloss(SgdLoss::Hinge, 2.0, 1.0, eps), 0.0);

    // SquaredHinge: z = 1 - p·y; z>0 → -2·y·z. p=0,y=1 → z=1 → -2.
    assert_eq!(dloss(SgdLoss::SquaredHinge, 0.0, 1.0, eps), -2.0);
    assert_eq!(dloss(SgdLoss::SquaredHinge, 2.0, 1.0, eps), 0.0);

    // Log: -y/(1+exp(y·p)). p=0,y=1 → -1/2.
    assert!((dloss(SgdLoss::Log, 0.0, 1.0, eps) - (-0.5)).abs() < 1e-12);

    // SquaredError: p - y. p=3,y=1 → 2.
    assert_eq!(dloss(SgdLoss::SquaredError, 3.0, 1.0, eps), 2.0);

    // EpsilonInsensitive: y-p>eps → -1; p-y>eps → 1; else 0.
    assert_eq!(dloss(SgdLoss::EpsilonInsensitive, 0.0, 1.0, eps), -1.0); // y-p=1>0.1
    assert_eq!(dloss(SgdLoss::EpsilonInsensitive, 2.0, 1.0, eps), 1.0); // p-y=1>0.1
    assert_eq!(dloss(SgdLoss::EpsilonInsensitive, 1.05, 1.0, eps), 0.0); // within eps

    // SquaredEpsilonInsensitive: z=y-p; z>eps → -2(z-eps); z<-eps → 2(-z-eps); else 0.
    // p=0,y=1 → z=1>0.1 → -2(0.9) = -1.8.
    assert!((dloss(SgdLoss::SquaredEpsilonInsensitive, 0.0, 1.0, eps) - (-1.8)).abs() < 1e-12);
    // p=2,y=1 → z=-1 < -0.1 → 2(1-0.1)=1.8.
    assert!((dloss(SgdLoss::SquaredEpsilonInsensitive, 2.0, 1.0, eps) - 1.8).abs() < 1e-12);
    assert_eq!(dloss(SgdLoss::SquaredEpsilonInsensitive, 1.05, 1.0, eps), 0.0);
}

/// Schedule isolation: the CONSTANT case is asserted FIRST (A1 / Pitfall 3) so a
/// loss/penalty bug is separable from a `t0`/`optimal` bug. Then invscaling, then
/// optimal (with the Bottou t0).
#[test]
fn schedule_constant_then_invscaling_then_optimal() {
    let alpha = 1e-4f64;
    let eta0 = 0.01f64;
    let power_t = 0.5f64;
    let t0 = optimal_t0(SgdLoss::Hinge, alpha);

    // CONSTANT — eta == eta0 regardless of t (the isolation case).
    assert_eq!(schedule_eta(SgdSchedule::Constant, 1, eta0, alpha, power_t, t0), eta0);
    assert_eq!(schedule_eta(SgdSchedule::Constant, 99, eta0, alpha, power_t, t0), eta0);

    // INVSCALING — eta = eta0 / t^power_t. t=4, power_t=0.5 → 0.01/2 = 0.005.
    let inv = schedule_eta(SgdSchedule::InvScaling, 4, eta0, alpha, power_t, t0);
    assert!((inv - 0.005).abs() < 1e-12, "invscaling eta={inv}");

    // OPTIMAL — eta(t) = 1/(alpha·(t0 + t - 1)). For hinge, dloss(-typw,1) = -1, so
    // initial_eta0 = typw and t0 = 1/(typw·alpha). Recompute the expected value.
    let typw = (1.0f64 / alpha.sqrt()).sqrt();
    let expected_t0 = 1.0 / (typw * alpha);
    assert!((t0 - expected_t0).abs() < 1e-6, "t0={t0} expected {expected_t0}");
    let opt = schedule_eta(SgdSchedule::Optimal, 1, eta0, alpha, power_t, t0);
    let expected = 1.0 / (alpha * (t0 + 1.0 - 1.0));
    assert!((opt - expected).abs() < 1e-9, "optimal eta={opt} expected {expected}");
}

// ===========================================================================
// sgd_convex_objective — the PRIM-10 standalone convex gate.
// ===========================================================================

/// Build a strongly-convex squared-error system `y = X·w* + b*` (no noise) with a
/// KNOWN closed-form minimizer, run `sgd_solve` with a constant schedule + near-
/// zero alpha to many epochs, and assert the iterate reaches the host optimum.
fn run_convex_objective<F: Float + CubeElement + bytemuck::Pod>(label: &str, tol: f64) {
    let (n, d) = (40usize, 3usize);
    // A well-conditioned design with column means ~0 so the unpenalized OLS optimum
    // is well-defined and SGD with a modest constant eta converges cleanly.
    let mut x = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            // Centered, bounded spread (deterministic).
            x[i * d + j] = (((i * d + j) % 7) as f64) * 0.3 - 0.9;
        }
    }
    let w_star = [1.3f64, -0.7, 0.5];
    let b_star = 0.4f64;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = b_star;
            for j in 0..d {
                acc += x[i * d + j] * w_star[j];
            }
            acc
        })
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_f: Vec<F> = x.iter().map(|&v| to_f::<F>(v)).collect();
    let y_f: Vec<F> = y.iter().map(|&v| to_f::<F>(v)).collect();
    let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, &x_f);
    let y_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, &y_f);

    let params = SgdParams {
        loss: SgdLoss::SquaredError,
        schedule: SgdSchedule::Constant,
        alpha: 1e-9, // near-zero L2 → recover the unpenalized OLS optimum.
        l1_ratio: 0.0,
        apply_l1: false,
        fit_intercept: true,
        eta0: 0.05,
        power_t: 0.5,
        epsilon: 0.1,
        batch_size: n, // full-batch → deterministic gradient descent.
        max_iter: 4000,
        tol: 0.0, // run all epochs (deterministic).
    };

    let (coef, intercept) =
        sgd_solve::<F>(&mut pool, &x_dev, &y_dev, (n, d), &params).expect("sgd_solve converges");

    let coef_h: Vec<f64> = coef.to_host(&pool).iter().map(|&v| from_f::<F>(v)).collect();
    let b_h = from_f::<F>(intercept.to_host(&pool)[0]);

    for j in 0..d {
        assert!(
            (coef_h[j] - w_star[j]).abs() <= tol,
            "[{label}] coef[{j}]={} != w*={} (tol {tol})",
            coef_h[j],
            w_star[j]
        );
    }
    assert!(
        (b_h - b_star).abs() <= tol,
        "[{label}] intercept={b_h} != b*={b_star} (tol {tol})"
    );

    coef.release_into(&mut pool);
    intercept.release_into(&mut pool);
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);
}

/// PRIM-10 standalone convex-objective gate (RESEARCH §Validation Criterion 1):
/// `sgd_solve` minimizes a known squared-error system to the host closed-form
/// optimum. f64 strict 1e-5; f32 a documented band.
#[test]
fn sgd_convex_objective() {
    let _ = env_logger::builder().is_test(true).try_init();

    // f32: documented round-off band (many-step accumulation on the flat surface).
    run_convex_objective::<f32>("f32", 1e-3);

    // f64: strict 1e-5 (runs on cpu, skips-with-log on rocm).
    if capability::skip_f64_with_log() {
        return;
    }
    run_convex_objective::<f64>("f64", 1e-5);
}

// ===========================================================================
// Device dloss (sgd_grad) — kernel-vs-host gate for every loss family.
// ===========================================================================

/// Launch `sgd_grad` over a `(p, y)` batch for one loss family and read `g[]`.
fn launch_grad<F: Float + CubeElement + bytemuck::Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    p: &[f64],
    y: &[f64],
    loss: SgdLoss,
    epsilon: f64,
) -> Vec<f64> {
    let b = p.len();
    let p_f: Vec<F> = p.iter().map(|&v| to_f::<F>(v)).collect();
    let y_f: Vec<F> = y.iter().map(|&v| to_f::<F>(v)).collect();
    let p_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &p_f);
    let y_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &y_f);
    let g_handle = pool.acquire(b * std::mem::size_of::<F>());

    let client = pool.client().clone();
    let block = 256u32;
    let count = CubeCount::Static(((b as u32) + block - 1) / block, 1, 1);
    let dim = CubeDim {
        x: block,
        y: 1,
        z: 1,
    };
    sgd_grad::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(p_dev.handle().clone(), b) },
        unsafe { ArrayArg::from_raw_parts(y_dev.handle().clone(), b) },
        unsafe { ArrayArg::from_raw_parts(g_handle.clone(), b) },
        0u32,
        b as u32,
        loss_id(loss),
        to_f::<F>(epsilon),
    );

    let g_dev = DeviceArray::<ActiveRuntime, F>::from_raw(g_handle.clone(), b);
    let host: Vec<f64> = g_dev.to_host(pool).iter().map(|&v| from_f::<F>(v)).collect();
    p_dev.release_into(pool);
    y_dev.release_into(pool);
    pool.release(g_handle, b * std::mem::size_of::<F>());
    host
}

fn run_grad_match<F: Float + CubeElement + bytemuck::Pod>(label: &str) {
    // A grid that exercises every branch of every loss: inside/outside the
    // hinge margin, both epsilon-tube sides, a boundary point (p·y == 1, the
    // duplicate-style degenerate case R-9 warns about), and both label signs.
    let p = [0.5f64, 2.0, -1.5, 1.0, 1.05, 0.0, -0.3, 3.0];
    let y = [1.0f64, 1.0, -1.0, 1.0, 1.0, -1.0, 0.4, 1.2];
    let eps = 0.1f64;

    let losses = [
        SgdLoss::Hinge,
        SgdLoss::Log,
        SgdLoss::SquaredHinge,
        SgdLoss::SquaredError,
        SgdLoss::EpsilonInsensitive,
        SgdLoss::SquaredEpsilonInsensitive,
    ];

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let tol = if std::mem::size_of::<F>() == 8 {
        1e-12
    } else {
        1e-5
    };
    for loss in losses {
        let dev = launch_grad::<F>(&mut pool, &p, &y, loss, eps);
        for i in 0..p.len() {
            let host = dloss(loss, p[i], y[i], eps).clamp(-1e12, 1e12);
            assert!(
                (dev[i] - host).abs() <= tol,
                "[{label}] sgd_grad {loss:?} g[{i}]={} != host dloss {} (p={}, y={}, tol {tol})",
                dev[i],
                host,
                p[i],
                y[i]
            );
        }
    }
}

/// The device `sgd_grad` dloss table matches the host [`dloss`] reference for
/// EVERY loss family and branch (VALUES asserted, not just non-panic — the
/// silent-miscompile discipline). f64 strict; f32 a round-off band.
#[test]
fn sgd_grad_matches_host_dloss() {
    let _ = env_logger::builder().is_test(true).try_init();
    run_grad_match::<f32>("f32");
    if capability::skip_f64_with_log() {
        return;
    }
    run_grad_match::<f64>("f64");
}

// ===========================================================================
// Device cumulative-L1 (sgd_l1_shrink) — kernel-vs-host gate.
// ===========================================================================

/// Host reference: the EXACT per-sample cumulative-L1 loop the previous host
/// implementation ran (samples outer, coordinates inner).
#[allow(clippy::too_many_arguments)]
fn host_l1_shrink(
    w: &mut [f64],
    q: &mut [f64],
    u_start: f64,
    du: f64,
    b: usize,
) -> f64 {
    let mut u = u_start;
    for _ in 0..b {
        u += du;
        for j in 0..w.len() {
            let z = w[j];
            if w[j] > 0.0 {
                w[j] = (w[j] - (u + q[j])).max(0.0);
            } else if w[j] < 0.0 {
                w[j] = (w[j] + (u - q[j])).min(0.0);
            }
            q[j] += w[j] - z;
        }
    }
    u
}

fn run_l1_shrink_match<F: Float + CubeElement + bytemuck::Pod>(label: &str) {
    // Mixed-sign weights incl. an exact zero (which the shrink must not move)
    // and small magnitudes that clip to zero under the budget.
    let w0 = [0.9f64, -0.6, 0.0, 0.004, -0.003, 1.4];
    let q0 = [0.01f64, -0.02, 0.0, 0.001, -0.001, 0.05];
    let (u_start, du, b) = (0.02f64, 0.005f64, 3usize);
    let d = w0.len();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let w_f: Vec<F> = w0.iter().map(|&v| to_f::<F>(v)).collect();
    let q_f: Vec<F> = q0.iter().map(|&v| to_f::<F>(v)).collect();
    let w_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, &w_f);
    let q_dev = DeviceArray::<ActiveRuntime, F>::from_host(&mut pool, &q_f);

    let cl = pool.client().clone();
    // One cube per coordinate, single unit (002-A selecting-unit shape).
    let count = CubeCount::Static(d as u32, 1, 1);
    let dim = CubeDim { x: 1, y: 1, z: 1 };
    sgd_l1_shrink::launch::<F, ActiveRuntime>(
        &cl,
        count,
        dim,
        unsafe { ArrayArg::from_raw_parts(w_dev.handle().clone(), d) },
        unsafe { ArrayArg::from_raw_parts(q_dev.handle().clone(), d) },
        to_f::<F>(u_start),
        to_f::<F>(du),
        d as u32,
        b as u32,
    );

    let mut w_ref = w0;
    let mut q_ref = q0;
    host_l1_shrink(&mut w_ref, &mut q_ref, u_start, du, b);

    let w_got: Vec<f64> = w_dev.to_host(&pool).iter().map(|&v| from_f::<F>(v)).collect();
    let q_got: Vec<f64> = q_dev.to_host(&pool).iter().map(|&v| from_f::<F>(v)).collect();

    let tol = if std::mem::size_of::<F>() == 8 {
        1e-12
    } else {
        1e-6
    };
    for j in 0..d {
        assert!(
            (w_got[j] - w_ref[j]).abs() <= tol,
            "[{label}] sgd_l1_shrink w[{j}]={} != host {} (tol {tol})",
            w_got[j],
            w_ref[j]
        );
        assert!(
            (q_got[j] - q_ref[j]).abs() <= tol,
            "[{label}] sgd_l1_shrink q[{j}]={} != host {} (tol {tol})",
            q_got[j],
            q_ref[j]
        );
    }

    w_dev.release_into(&mut pool);
    q_dev.release_into(&mut pool);
}

/// The device cumulative-L1 soft-shrink replays the exact host per-sample
/// sequence (w AND q asserted — the q correction is the part a re-ordering bug
/// silently corrupts). f64 strict; f32 a round-off band.
#[test]
fn sgd_l1_shrink_matches_host() {
    let _ = env_logger::builder().is_test(true).try_init();
    run_l1_shrink_match::<f32>("f32");
    if capability::skip_f64_with_log() {
        return;
    }
    run_l1_shrink_match::<f64>("f64");
}

// ===========================================================================
// tol > 0 early stop — the device-tracked convergence gate.
// ===========================================================================

/// With `tol > 0` on the convex system, the device-tracked stopping gate
/// (`sgd_copy` snapshot + `sgd_delta_max` epoch fold + one per-epoch readback)
/// must fire and still land near the optimum — the early-stop path is
/// exercised end-to-end (it has no other coverage; the oracle fixtures pin
/// `tol = 0`).
#[test]
fn sgd_tol_early_stop_converges() {
    let _ = env_logger::builder().is_test(true).try_init();

    let (n, d) = (40usize, 3usize);
    let mut x = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = (((i * d + j) % 7) as f64) * 0.3 - 0.9;
        }
    }
    let w_star = [1.3f64, -0.7, 0.5];
    let b_star = 0.4f64;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let mut acc = b_star;
            for j in 0..d {
                acc += x[i * d + j] * w_star[j];
            }
            acc
        })
        .collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_f: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let x_dev = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &x_f);
    let y_dev = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &y_f);

    let params = SgdParams {
        loss: SgdLoss::SquaredError,
        schedule: SgdSchedule::Constant,
        alpha: 1e-9,
        l1_ratio: 0.0,
        apply_l1: false,
        fit_intercept: true,
        eta0: 0.05,
        power_t: 0.5,
        epsilon: 0.1,
        batch_size: n,
        max_iter: 4000,
        tol: 1e-7, // small but > 0: the early stop fires near the fixed point.
    };

    let (coef, intercept) =
        sgd_solve::<f32>(&mut pool, &x_dev, &y_dev, (n, d), &params).expect("sgd_solve runs");

    let coef_h: Vec<f64> = coef.to_host(&pool).iter().map(|&v| v as f64).collect();
    let b_h = intercept.to_host(&pool)[0] as f64;
    for j in 0..d {
        assert!(
            (coef_h[j] - w_star[j]).abs() <= 1e-2,
            "early-stop coef[{j}]={} != w*={} (band 1e-2)",
            coef_h[j],
            w_star[j]
        );
    }
    assert!(
        (b_h - b_star).abs() <= 1e-2,
        "early-stop intercept={b_h} != b*={b_star} (band 1e-2)"
    );

    coef.release_into(&mut pool);
    intercept.release_into(&mut pool);
    x_dev.release_into(&mut pool);
    y_dev.release_into(&mut pool);
}
