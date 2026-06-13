//! Plan 05-06 — L-BFGS primitive standalone oracle.
//!
//! THE highest correctness risk in the project, validated in two stages
//! (RESEARCH Pitfall 5):
//!
//! 1. **Convex-quadratic invariant FIRST** — minimize `f(x) = ½xᵀAx − bᵀx`
//!    (gradient `Ax − b`) for a small SPD `A` and assert the final iterate equals
//!    the unique minimizer `x* = A⁻¹b` within 1e-5. The convex objective has a
//!    unique global minimum, so the iterate must match `A⁻¹b` regardless of small
//!    line-search differences — this isolates "is my L-BFGS correct" from "does it
//!    match sklearn's path" (the algebraic-invariant pattern of `cholesky_test`).
//!
//! 2. **Softmax loss/grad oracle** — the stable-softmax `softmax_loss_grad`
//!    launcher reproduces the RESEARCH multinomial objective `(1/n)Σloss +
//!    ½·l2_reg·‖coef‖²` (intercept unpenalized, `l2_reg = 1/(C·n)`) within 1e-5
//!    for BOTH the binary and multiclass fixtures, gradient cross-checked against a
//!    central finite difference; then a smoke test that `lbfgs_minimize` driven by
//!    `softmax_loss_grad` converges on the binary fixture without NaN (full
//!    sklearn `coef_` agreement is the ESTIMATOR's gate — plan 05-10).
//!
//! f64 functions carry the `skip_f64_with_log` gate (cpu runs f64; rocm skips,
//! D-07). Per AGENTS.md §2 tests live here, never an in-source `#[cfg(test)] mod
//! tests`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::lbfgs::{
    lbfgs_minimize, softmax_loss_grad, LBFGS_FTOL, LBFGS_GTOL, LBFGS_MAXITER, LBFGS_MAXLS,
};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// LogReg fixture geometry (gen_oracle.py LOG_N_SAMPLES × LOG_N_FEATURES; the
/// query/predict arrays use LOG_N_QUERY but the loss/grad oracle uses the train
/// design X/y).
const LOG_N_SAMPLES: usize = 40;
const LOG_N_FEATURES: usize = 4;
const LOG_C: f64 = 1.0;

/// 1e-5 oracle contract.
const TOL: f64 = 1e-5;

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

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("lbfgs tests are f32/f64 only"),
    }
}

/// Solve the small SPD system `A x = b` by Gaussian elimination with partial
/// pivoting (host reference for the convex-quadratic minimizer `x* = A⁻¹b`).
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Augmented [A | b].
    let mut m = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            m[i * (n + 1) + j] = a[i * n + j];
        }
        m[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = m[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = m[r * (n + 1) + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            for j in 0..(n + 1) {
                m.swap(col * (n + 1) + j, piv * (n + 1) + j);
            }
        }
        let diag = m[col * (n + 1) + col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r * (n + 1) + col] / diag;
            for j in col..(n + 1) {
                m[r * (n + 1) + j] -= factor * m[col * (n + 1) + j];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        x[i] = m[i * (n + 1) + n] / m[i * (n + 1) + i];
    }
    x
}

// ===========================================================================
// Stage 1 — convex-quadratic standalone invariant (Pitfall 5).
// ===========================================================================

/// Run the convex-quadratic minimizer `½xᵀAx − bᵀx → x* = A⁻¹b` invariant for a
/// fixed small SPD `A` and assert the L-BFGS iterate matches `A⁻¹b` within 1e-5.
fn check_convex_quadratic() {
    // A small, well-conditioned SPD matrix (diagonally dominant) and a target b.
    let n = 4usize;
    #[rustfmt::skip]
    let a: Vec<f64> = vec![
        4.0, 1.0, 0.0, 0.5,
        1.0, 3.0, 0.5, 0.0,
        0.0, 0.5, 2.5, 1.0,
        0.5, 0.0, 1.0, 3.5,
    ];
    let b: Vec<f64> = vec![1.0, -2.0, 0.5, 3.0];

    // Reference minimizer x* = A⁻¹b (host Gaussian elimination).
    let x_star = solve_spd(&a, &b, n);

    // f(x) = ½ xᵀA x − bᵀx ; grad = A x − b.
    let a_cl = a.clone();
    let b_cl = b.clone();
    let f = move |x: &[f64]| -> (f64, Vec<f64>) {
        // Ax.
        let mut ax = vec![0.0f64; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += a_cl[i * n + j] * x[j];
            }
            ax[i] = s;
        }
        // loss = ½ xᵀ(Ax) − bᵀx ; grad = Ax − b.
        let mut quad = 0.0;
        let mut bx = 0.0;
        let mut grad = vec![0.0f64; n];
        for i in 0..n {
            quad += x[i] * ax[i];
            bx += b_cl[i] * x[i];
            grad[i] = ax[i] - b_cl[i];
        }
        (0.5 * quad - bx, grad)
    };

    let x0 = vec![0.0f64; n];
    let res = lbfgs_minimize(x0, f, LBFGS_GTOL, LBFGS_FTOL, LBFGS_MAXLS, LBFGS_MAXITER)
        .expect("lbfgs_minimize convex quadratic");

    assert!(
        res.converged,
        "L-BFGS should converge on the convex quadratic (max_grad {:e}, iters {})",
        res.max_grad, res.iters
    );
    for i in 0..n {
        let abs = (res.x[i] - x_star[i]).abs();
        assert!(
            abs <= TOL,
            "convex-quadratic minimizer mismatch at x[{i}]: got {}, want {} (abs {abs:e})",
            res.x[i],
            x_star[i]
        );
    }
}

/// Standalone convex-quadratic minimizer invariant `x* = A⁻¹b` within 1e-5, f32.
/// (The solver is dtype-agnostic — it runs the host loop in f64 — but the gate
/// schedules an f32-labelled and an f64-labelled case for parity with the other
/// oracles.)
#[test]
fn lbfgs_convex_quadratic_minimizer_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_convex_quadratic();
}

/// Standalone convex-quadratic minimizer invariant, f64 (cpu runs; rocm skips).
#[test]
fn lbfgs_convex_quadratic_minimizer_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_convex_quadratic();
}

// ===========================================================================
// Stage 2 — softmax loss/grad oracle (binary + multiclass) + lbfgs smoke.
// ===========================================================================

/// Host reference for the RESEARCH multinomial objective at `(w, b)`:
/// `loss = (1/n)Σ_i(lse[i] − raw[i,y_i]) + ½·l2_reg·‖w‖²` (intercept unpenalized).
/// Returns just the scalar loss (used by the finite-difference gradient check).
fn ref_loss(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    b: &[f64],
    n: usize,
    d: usize,
    k: usize,
    l2_reg: f64,
) -> f64 {
    let mut loss = 0.0f64;
    for i in 0..n {
        // raw[i,k] and row max.
        let mut raw = vec![0.0f64; k];
        for c in 0..k {
            let mut s = b[c];
            for j in 0..d {
                s += x[i * d + j] * w[c * d + j];
            }
            raw[c] = s;
        }
        let m = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0;
        for c in 0..k {
            sum_exp += (raw[c] - m).exp();
        }
        let lse = m + sum_exp.ln();
        let yi = y[i] as usize;
        loss += lse - raw[yi];
    }
    loss /= n as f64;
    // ½·l2_reg·‖w‖² (intercept unpenalized).
    let mut wn2 = 0.0;
    for &v in w {
        wn2 += v * v;
    }
    loss + 0.5 * l2_reg * wn2
}

/// Drive `softmax_loss_grad` on a fixture at a fixed `(w, b)` test point and
/// assert (a) the device loss matches the host reference within 1e-5 and (b) the
/// device gradient matches a central finite difference of the host reference.
fn check_softmax_oracle<F>(fixture_name: &str, k: usize)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load logistic fixture");
    let x_raw: Vec<f64> = case.expect_f64("X").to_vec();
    let y_raw: Vec<f64> = case.expect_f64("y").to_vec();
    // The multiclass blob keeps `per = LOG_N_SAMPLES // n_classes` rows per class,
    // so n = y.len() (39 for K=3, 40 for K=2) — derive it, don't hardcode.
    let d = LOG_N_FEATURES;
    let n = y_raw.len();
    assert_eq!(x_raw.len(), n * d, "X geometry");
    assert_eq!(y_raw.len(), n, "y geometry");

    let l2_reg = 1.0 / (LOG_C * n as f64); // Pitfall 3: l2_reg = 1/(C·n).

    // Fixed, reproducible test point (small deterministic values, not all-zero so
    // the gradient is non-trivial).
    let mut w = vec![0.0f64; k * d];
    for (idx, wv) in w.iter_mut().enumerate() {
        *wv = 0.1 * ((idx % 5) as f64) - 0.2;
    }
    let mut b = vec![0.0f64; k];
    for (c, bv) in b.iter_mut().enumerate() {
        *bv = 0.05 * (c as f64) - 0.1;
    }

    // Upload + launch the device kernel.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_f: Vec<F> = x_raw.iter().map(|&v| from_f64::<F>(v)).collect();
    let y_f: Vec<F> = y_raw.iter().map(|&v| from_f64::<F>(v)).collect();
    let w_f: Vec<F> = w.iter().map(|&v| from_f64::<F>(v)).collect();
    let b_f: Vec<F> = b.iter().map(|&v| from_f64::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);
    let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &w_f);
    let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b_f);

    let (loss, grad_w, grad_b) =
        softmax_loss_grad::<F>(&mut pool, &x_dev, &y_dev, &w_dev, &b_dev, n, d, k, l2_reg)
            .expect("softmax_loss_grad");

    // (a) Loss matches the host reference within 1e-5.
    let want_loss = ref_loss(&x_raw, &y_raw, &w, &b, n, d, k, l2_reg);
    let labs = (loss - want_loss).abs();
    assert!(
        labs <= TOL || labs / want_loss.abs().max(1e-12) <= TOL,
        "{fixture_name} softmax loss mismatch: got {loss}, want {want_loss} (abs {labs:e})"
    );
    assert!(loss.is_finite(), "{fixture_name} softmax loss is not finite");

    // (b) gradW matches a central finite difference of the host reference.
    let h = 1e-6;
    for idx in 0..(k * d) {
        let mut wp = w.clone();
        let mut wm = w.clone();
        wp[idx] += h;
        wm[idx] -= h;
        let fp = ref_loss(&x_raw, &y_raw, &wp, &b, n, d, k, l2_reg);
        let fm = ref_loss(&x_raw, &y_raw, &wm, &b, n, d, k, l2_reg);
        let fd = (fp - fm) / (2.0 * h);
        let abs = (grad_w[idx] - fd).abs();
        assert!(
            abs <= 1e-5 || abs / fd.abs().max(1e-12) <= 1e-4,
            "{fixture_name} gradW[{idx}] mismatch: device {}, FD {} (abs {abs:e})",
            grad_w[idx],
            fd
        );
    }
    // gradb matches a central finite difference (perturb b).
    for c in 0..k {
        let mut bp = b.clone();
        let mut bm = b.clone();
        bp[c] += h;
        bm[c] -= h;
        let fp = ref_loss(&x_raw, &y_raw, &w, &bp, n, d, k, l2_reg);
        let fm = ref_loss(&x_raw, &y_raw, &w, &bm, n, d, k, l2_reg);
        let fd = (fp - fm) / (2.0 * h);
        let abs = (grad_b[c] - fd).abs();
        assert!(
            abs <= 1e-5 || abs / fd.abs().max(1e-12) <= 1e-4,
            "{fixture_name} gradb[{c}] mismatch: device {}, FD {} (abs {abs:e})",
            grad_b[c],
            fd
        );
    }
}

/// LOAD-NOT-JUST-PRESENT: BOTH the `logistic_binary` and `logistic_multi`
/// fixtures load with well-formed coef/intercept arrays.
#[test]
fn fixture_loads() {
    let bin = load_npz(fixture("logistic_binary_f64_seed42.npz")).expect("load logistic_binary_f64");
    assert_len(&bin, "coef", LOG_N_FEATURES);
    assert_len(&bin, "intercept", 1);
    assert_len(&bin, "X", LOG_N_SAMPLES * LOG_N_FEATURES);
    assert_len(&bin, "y", LOG_N_SAMPLES);

    let multi = load_npz(fixture("logistic_multi_f64_seed42.npz")).expect("load logistic_multi_f64");
    assert_len(&multi, "coef", 3 * LOG_N_FEATURES);
    assert_len(&multi, "intercept", 3);
}

/// Stable softmax loss/grad reproduces the reference (binary, K=2), f32.
#[test]
fn lbfgs_softmax_loss_grad_binary_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_softmax_oracle::<f32>("logistic_binary_f32_seed42.npz", 2);
}

/// Stable softmax loss/grad reproduces the reference (binary, K=2), f64.
#[test]
fn lbfgs_softmax_loss_grad_binary_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_softmax_oracle::<f64>("logistic_binary_f64_seed42.npz", 2);
}

/// Stable softmax loss/grad reproduces the reference (multiclass, K=3), f32.
#[test]
fn lbfgs_softmax_loss_grad_multi_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_softmax_oracle::<f32>("logistic_multi_f32_seed42.npz", 3);
}

/// Stable softmax loss/grad reproduces the reference (multiclass, K=3), f64.
#[test]
fn lbfgs_softmax_loss_grad_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_softmax_oracle::<f64>("logistic_multi_f64_seed42.npz", 3);
}

/// Smoke test: `lbfgs_minimize` driven by `softmax_loss_grad` converges on the
/// binary fixture (max |grad| under gtol) without NaN — full sklearn `coef_`
/// agreement is the ESTIMATOR's gate (plan 05-10).
fn check_lbfgs_softmax_smoke<F>(fixture_name: &str, k: usize)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load logistic fixture");
    let x_raw: Vec<f64> = case.expect_f64("X").to_vec();
    let y_raw: Vec<f64> = case.expect_f64("y").to_vec();
    let d = LOG_N_FEATURES;
    let n = y_raw.len();
    let l2_reg = 1.0 / (LOG_C * n as f64);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_f: Vec<F> = x_raw.iter().map(|&v| from_f64::<F>(v)).collect();
    let y_f: Vec<F> = y_raw.iter().map(|&v| from_f64::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_f);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);

    // The parameter vector is [W (k×d) | b (k)] flattened; the closure splits it,
    // launches the kernel, and re-flattens (gradW | gradb).
    let x0 = vec![0.0f64; k * d + k];
    let f = |params: &[f64]| -> (f64, Vec<f64>) {
        let w: Vec<F> = params[..k * d].iter().map(|&v| from_f64::<F>(v)).collect();
        let b: Vec<F> = params[k * d..].iter().map(|&v| from_f64::<F>(v)).collect();
        let w_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &w);
        let b_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &b);
        let (loss, grad_w, grad_b) =
            softmax_loss_grad::<F>(&mut pool, &x_dev, &y_dev, &w_dev, &b_dev, n, d, k, l2_reg)
                .expect("softmax_loss_grad");
        w_dev.release_into(&mut pool);
        b_dev.release_into(&mut pool);
        let mut grad = Vec::with_capacity(k * d + k);
        grad.extend_from_slice(&grad_w);
        grad.extend_from_slice(&grad_b);
        (loss, grad)
    };

    let res = lbfgs_minimize(x0, f, LBFGS_GTOL, LBFGS_FTOL, LBFGS_MAXLS, LBFGS_MAXITER)
        .expect("lbfgs_minimize softmax");

    assert!(res.loss.is_finite(), "{fixture_name} L-BFGS loss is NaN/Inf");
    for v in &res.x {
        assert!(v.is_finite(), "{fixture_name} L-BFGS iterate has NaN/Inf");
    }
    assert!(
        res.converged,
        "{fixture_name} L-BFGS+softmax should converge (max_grad {:e}, iters {})",
        res.max_grad, res.iters
    );
}

/// L-BFGS + softmax converges on the binary fixture without NaN, f32.
#[test]
fn lbfgs_softmax_converges_binary_f32() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    check_lbfgs_softmax_smoke::<f32>("logistic_binary_f32_seed42.npz", 2);
}

/// L-BFGS + softmax converges on the binary fixture without NaN, f64 (cpu runs;
/// rocm skips, D-07).
#[test]
fn lbfgs_softmax_converges_binary_f64() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        return;
    }
    check_lbfgs_softmax_smoke::<f64>("logistic_binary_f64_seed42.npz", 2);
}
