//! Where does a stacking fold sweep actually spend its time? (STACK-FOLD-01)
//!
//! `StackingRegressor` with an int `cv = k` runs `k` out-of-fold fits per member
//! plus one full fit. Today each of those fold fits is driven from Python: the
//! shim fancy-indexes `X[train]` in numpy, normalizes it into an Arrow buffer,
//! and uploads the result — so the design crosses the bus `k` times per member
//! and is copied on the host twice per crossing.
//!
//! This probe A/Bs that shape against the one the Rust engine can do instead —
//! upload `X` and `y` ONCE, then gather each fold's rows on the device — with
//! everything else held identical:
//!
//! | arm | per fold |
//! |---|---|
//! | `reupload` | host gather → `DeviceArray::from_host` → fit → host gather → predict |
//! | `resident` | `gather_rows_device` → fit → `gather_rows_device` → predict |
//!
//! Both arms run the SAME `Ridge` solver on the SAME rows, and the probe asserts
//! their coefficients agree before reporting any time — a ladder between two
//! arms that computed different things is worthless.
//!
//! The timer includes the upload in both arms, because a Python caller passes a
//! numpy array and the transfer is part of `fit` for them (the
//! `bayesian_ridge_perf_test` rule). The `resident` arm's single upload is
//! inside its timer too, so the comparison is honest end to end.
//!
//! ## Dtype
//! Run at the dtype the backend actually uses for a stack: the shim's
//! `pick_dtype` sends f32 to an f64-incapable backend, so measuring f64 there
//! would measure a configuration no user reaches. rocm additionally has no f64
//! GEMM, so a f64 Ridge device fit cannot run there at all.
//!
//! ```text
//! cargo test -p mlrs-algos --release --features rocm \
//!   --test stacking_folds_perf_test -- --ignored --nocapture
//! ```
//!
//! Per AGENTS.md §2 this lives in `tests/`, never in `src/`.

use std::time::{Duration, Instant};

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ridge::Ridge;
use mlrs_algos::typestate::Predict;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::kmeans::gather_rows_device;
use mlrs_backend::runtime::{self, ActiveRuntime};

const SEED: u64 = 42;

/// A deterministic well-conditioned design, host-side, in f64.
fn design(n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state = SEED;
    let mut next = || {
        // xorshift64* — deterministic and dependency-free; the numbers only
        // have to be non-degenerate, not statistically pristine.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let x: Vec<f64> = (0..n * d).map(|_| next()).collect();
    let w: Vec<f64> = (0..d).map(|_| next()).collect();
    let y: Vec<f64> = (0..n)
        .map(|r| (0..d).map(|c| x[r * d + c] * w[c]).sum::<f64>())
        .collect();
    (x, y)
}

fn to_f<F: Float + CubeElement + Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("probe is f32/f64 only"),
    }
}

fn from_f<F: Float + CubeElement + Pod>(v: &F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(v)),
        _ => unreachable!("probe is f32/f64 only"),
    }
}

/// `k` contiguous KFold-style splits as `(train_idx, test_idx)`.
fn folds(n: usize, k: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let per = n / k;
    (0..k)
        .map(|f| {
            let lo = f * per;
            let hi = if f + 1 == k { n } else { (f + 1) * per };
            let test: Vec<u32> = (lo..hi).map(|i| i as u32).collect();
            let train: Vec<u32> = (0..n as u32)
                .filter(|i| !(lo..hi).contains(&(*i as usize)))
                .collect();
            (train, test)
        })
        .collect()
}

fn host_gather<F: Copy + Default>(x: &[F], idx: &[u32], d: usize) -> Vec<F> {
    let mut out = vec![F::default(); idx.len() * d];
    for (r, &i) in idx.iter().enumerate() {
        out[r * d..(r + 1) * d].copy_from_slice(&x[i as usize * d..(i as usize + 1) * d]);
    }
    out
}

/// TODAY's shape: every fold re-uploads its own slice of the design.
fn arm_reupload<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    y: &[F],
    (_n, d): (usize, usize),
    splits: &[(Vec<u32>, Vec<u32>)],
) -> (Duration, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let t0 = Instant::now();
    let mut coefs = Vec::new();
    for (train, test) in splits {
        let xt = host_gather(x, train, d);
        let yt: Vec<F> = train.iter().map(|&i| y[i as usize]).collect();
        let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &xt);
        let y_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &yt);

        let fitted = Ridge::<F>::new()
            .fit_with_sample_weight(pool, &x_dev, Some(&y_dev), (train.len(), d), None)
            .expect("ridge fit");
        coefs.push(from_f(&fitted.coef(pool)[0]));

        let xs = host_gather(x, test, d);
        let _ = fitted
            .predict_from_host(pool, &xs, (test.len(), d))
            .expect("ridge predict");
        x_dev.release_into(pool);
        y_dev.release_into(pool);
    }
    (t0.elapsed(), coefs)
}

/// The PROPOSED shape: one upload, per-fold gathers on the device.
fn arm_resident<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    y: &[F],
    (n, d): (usize, usize),
    splits: &[(Vec<u32>, Vec<u32>)],
) -> (Duration, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let t0 = Instant::now();
    let x_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, x);
    let y_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, y);

    let mut coefs = Vec::new();
    for (train, test) in splits {
        let xt = gather_rows_device::<F>(pool, &x_dev, train, n, d).expect("gather train x");
        let yt = gather_rows_device::<F>(pool, &y_dev, train, n, 1).expect("gather train y");

        let fitted = Ridge::<F>::new()
            .fit_with_sample_weight(pool, &xt, Some(&yt), (train.len(), d), None)
            .expect("ridge fit");
        coefs.push(from_f(&fitted.coef(pool)[0]));

        let xs = gather_rows_device::<F>(pool, &x_dev, test, n, d).expect("gather test x");
        let out = fitted
            .predict(pool, &xs, (test.len(), d))
            .expect("ridge predict");
        out.release_into(pool);
        xt.release_into(pool);
        yt.release_into(pool);
        xs.release_into(pool);
    }
    x_dev.release_into(pool);
    y_dev.release_into(pool);
    (t0.elapsed(), coefs)
}

fn run_ladder<F>(label: &str)
where
    F: Float + CubeElement + Pod,
{
    let backend = mlrs_backend::capability::active_backend_name();
    println!("\n=== stacking fold sweep, backend={backend}, {label}, min of 3 ===");
    println!(
        "{:<26}{:>12}{:>12}{:>10}",
        "shape", "reupload", "resident", "speedup"
    );

    for &(n, d, k) in &[
        (20_000usize, 32usize, 5usize),
        (100_000, 32, 5),
        (100_000, 64, 5),
        (100_000, 64, 10),
        (200_000, 64, 5),
    ] {
        let (x64, y64) = design(n, d);
        let x: Vec<F> = x64.iter().copied().map(to_f).collect();
        let y: Vec<F> = y64.iter().copied().map(to_f).collect();
        let splits = folds(n, k);
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

        // Warm both arms once (kernel JIT, pool growth) before timing either,
        // and check they agree — a ladder between arms that fitted different
        // models would be meaningless.
        let (_, warm_a) = arm_reupload::<F>(&mut pool, &x, &y, (n, d), &splits);
        let (_, warm_b) = arm_resident::<F>(&mut pool, &x, &y, (n, d), &splits);
        for (a, b) in warm_a.iter().zip(&warm_b) {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "the two arms must fit the SAME model: {a} vs {b}"
            );
        }

        let mut best_a = Duration::MAX;
        let mut best_b = Duration::MAX;
        for _ in 0..3 {
            // Interleaved, not blocked, so a drifting machine costs both arms
            // the same (mlrs-cpu-bench-separate-processes).
            best_a = best_a.min(arm_reupload::<F>(&mut pool, &x, &y, (n, d), &splits).0);
            best_b = best_b.min(arm_resident::<F>(&mut pool, &x, &y, (n, d), &splits).0);
        }

        let a_ms = best_a.as_secs_f64() * 1e3;
        let b_ms = best_b.as_secs_f64() * 1e3;
        println!(
            "{:<26}{:>11.1}m{:>11.1}m{:>9.2}x",
            format!("n={n} d={d} k={k}"),
            a_ms,
            b_ms,
            a_ms / b_ms
        );
    }
}

#[test]
#[ignore = "perf probe — run explicitly with --ignored --nocapture"]
fn fold_sweep_reupload_vs_resident() {
    let _ = env_logger::builder().is_test(true).try_init();
    run_ladder::<f32>("f32");
    // f64 only where the backend can actually solve in it: rocm/cuda have no
    // f64 GEMM, and the shim would have sent f32 there anyway.
    if !mlrs_backend::capability::skip_f64_with_log() {
        run_ladder::<f64>("f64");
    }
}
