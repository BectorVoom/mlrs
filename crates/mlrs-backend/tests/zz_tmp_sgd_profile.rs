//! TEMPORARY ingress/solve attribution probe (deleted before hand-off).
use std::time::Instant;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::sgd::{sgd_solve, SgdLoss, SgdParams, SgdSchedule};
use mlrs_backend::runtime::{self, ActiveRuntime};

#[test]
fn profile_sgd_phases() {
    let (n, d, iters) = (50_000usize, 64usize, 5usize);
    let mut s = 12345u64;
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let mut t = 0.0f32;
        for j in 0..d {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((s >> 33) as f32 / (1u32 << 31) as f32) - 0.5;
            x.push(v);
            if j < 2 {
                t += v;
            }
        }
        y.push(if t >= 0.0 { 1.0f32 } else { -1.0 });
    }

    let params = SgdParams {
        loss: SgdLoss::Hinge,
        schedule: SgdSchedule::Optimal,
        alpha: 1e-4,
        l1_ratio: 0.15,
        apply_l1: false,
        fit_intercept: true,
        eta0: 0.01,
        power_t: 0.5,
        epsilon: 0.0,
        batch_size: 1,
        max_iter: iters,
        tol: 0.0,
    };

    let client = runtime::active_client();
    let mut pool = BufferPool::<ActiveRuntime>::new(client);

    for rep in 0..4 {
        let t0 = Instant::now();
        let x_dev = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &x);
        let y_dev = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &y);
        let t_up = t0.elapsed();

        let t1 = Instant::now();
        let xh = x_dev.to_host(&mut pool);
        let t_down = t1.elapsed();
        std::hint::black_box(&xh);

        let t2 = Instant::now();
        let (c, b) = sgd_solve::<f32>(&mut pool, &x_dev, &y_dev, (n, d), &params).unwrap();
        let t_solve = t2.elapsed();

        let t3 = Instant::now();
        let ch = c.to_host(&mut pool);
        let t_out = t3.elapsed();
        std::hint::black_box(&ch);

        eprintln!(
            "rep{rep}: upload={:?} readback={:?} solve(incl. its own to_host)={:?} out={:?}",
            t_up, t_down, t_solve, t_out
        );
        c.release_into(&mut pool);
        b.release_into(&mut pool);
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
    }
}
