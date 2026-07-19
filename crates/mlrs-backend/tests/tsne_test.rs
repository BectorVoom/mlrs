//! TSNE-01 — the exact-method t-SNE gradient prim VALUE gate.
//!
//! `tsne_gradient` composes the Phase-2 `distance(sqrt=false)` prim with the
//! two new kernels (`tsne_qnum` Student-t affinity + `tsne_grad` KL-gradient
//! GATHER). This harness evaluates it under the concrete `ActiveRuntime` on a
//! small random embedding and asserts the returned gradient, `qsum`, and the
//! device-resident `qnum` block against an in-test HOST reference walk of
//! sklearn's `_kl_divergence` (a happy-path non-panic check would miss a
//! mis-lowered GATHER — the mutual_reachability_test.rs convention).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate (cpu runs f64; rocm
//! skips). Per AGENTS.md §2 tests live here, never an in-source
//! `#[cfg(test)] mod tests`.

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::tsne::{tsne_gradient, MACHINE_EPSILON};
use mlrs_backend::runtime::{self, ActiveRuntime};

fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("tsne tests are f32/f64 only"),
    }
}

fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("tsne tests are f32/f64 only"),
    }
}

/// Host reference: sklearn `_kl_divergence` internals at `dof = 1` —
/// `qnum[i,j] = 1/(1+‖y_i−y_j‖²)` (diag 0), `qsum = Σ qnum`,
/// `grad[i,c] = 4·Σ_j (p_ij − max(qnum_ij/qsum, eps))·qnum_ij·(y_ic − y_jc)`.
fn reference(y: &[f64], p: &[f64], n: usize, d: usize) -> (Vec<f64>, f64, Vec<f64>) {
    let mut qnum = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut dsq = 0.0;
                for c in 0..d {
                    let diff = y[i * d + c] - y[j * d + c];
                    dsq += diff * diff;
                }
                qnum[i * n + j] = 1.0 / (1.0 + dsq);
            }
        }
    }
    let qsum: f64 = qnum.iter().sum();
    let mut grad = vec![0.0f64; n * d];
    for i in 0..n {
        for c in 0..d {
            let mut acc = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let q = (qnum[i * n + j] / qsum).max(MACHINE_EPSILON);
                acc += (p[i * n + j] - q) * qnum[i * n + j] * (y[i * d + c] - y[j * d + c]);
            }
            grad[i * d + c] = 4.0 * acc;
        }
    }
    (qnum, qsum, grad)
}

fn run_case<F>(tol: f64)
where
    F: cubecl::prelude::Float + cubecl::prelude::CubeElement + bytemuck::Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Small deterministic embedding + a valid-ish P (row-normalized softmax
    // of negated distances — exact P semantics are irrelevant to the VALUE
    // gate, only the arithmetic contract is).
    let n = 7usize;
    let d = 2usize;
    let y: Vec<f64> = (0..n * d)
        .map(|k| ((k * 2654435761 % 97) as f64) / 17.0 - 2.5)
        .collect();
    let mut p = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                p[i * n + j] = 1.0 / (n as f64 * (n as f64 - 1.0));
            }
        }
    }

    let (qnum_ref, qsum_ref, grad_ref) = reference(&y, &p, n, d);

    let y_f: Vec<F> = y.iter().map(|&v| from_f64::<F>(v)).collect();
    let p_f: Vec<F> = p.iter().map(|&v| from_f64::<F>(v)).collect();
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y_f);
    let p_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &p_f);

    let step = tsne_gradient::<F>(&mut pool, &y_dev, &p_dev, n, d, 1.0)
        .expect("tsne_gradient on a valid geometry");

    let qnum_got: Vec<f64> = step.qnum.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    for (g, r) in qnum_got.iter().zip(qnum_ref.iter()) {
        assert!((g - r).abs() <= tol, "qnum mismatch: got {g}, ref {r}");
    }
    assert!(
        (step.qsum - qsum_ref).abs() <= tol * n as f64 * n as f64,
        "qsum mismatch: got {}, ref {qsum_ref}",
        step.qsum
    );
    for (g, r) in step.grad.iter().zip(grad_ref.iter()) {
        let g = host_to_f64(*g);
        assert!((g - r).abs() <= tol, "grad mismatch: got {g}, ref {r}");
    }
    step.qnum.release_into(&mut pool);

    // Geometry violations are typed errors BEFORE any launch.
    assert!(tsne_gradient::<F>(&mut pool, &y_dev, &p_dev, n, 3, 1.0).is_err());
    assert!(tsne_gradient::<F>(&mut pool, &y_dev, &y_dev, n, d, 1.0).is_err());
}

#[test]
fn tsne_gradient_values_f32() {
    run_case::<f32>(1e-4);
}

#[test]
fn tsne_gradient_values_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    run_case::<f64>(1e-9);
}
