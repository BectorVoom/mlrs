//! Huber primal objective evaluator (`prims::huber_objective::HuberObjective`)
//! validation — HUBER-01.
//!
//! `HuberObjective` is what `HuberRegressor`'s L-BFGS solve calls per iteration
//! and line-search step: ONE pass over the design producing the margin-derived
//! gradient `x̃ᵀ·g` plus the three scalar reductions the `σ` derivative needs.
//! Its cpu arm fuses all five into a single `-O3` host pass split across a
//! persistent worker pool, where scikit-learn's NumPy form walks the design five
//! times and fancy-index COPIES it twice.
//!
//! Because that arm is an INDEPENDENT implementation of the same maths rather
//! than a tuned kernel, every test here checks it against a DIRECT, deliberately
//! naive host reference that builds the augmented design explicitly and runs
//! textbook loops. On top of that, [`gradient_matches_central_differences`]
//! checks the assembled objective's gradient — including `∂L/∂σ`, the entry no
//! reference implementation would catch a sign error in — against central
//! differences of the loss itself. That is the check that does not share a line
//! of code with the thing it validates.
//!
//! The cases cover both intercept modes, both dtypes, the weighted and
//! unweighted monomorphizations, the all-inlier / all-outlier boundary
//! behaviour, and the multi-worker split (the fan-out only engages above
//! `HUBER_ELEMS_PER_UNIT`, so one case is sized to force it).
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::capability::{self, FloatKind};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::huber_objective::{HuberDesign, HuberEval, HuberObjective};
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{f64_to_host, PrimError};

/// Deterministic `[-1, 1)` stream (splitmix64), so a failure is reproducible and
/// the two dtypes see the SAME values.
fn uniform_pm1(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// The naive reference: build `x̃` explicitly, then five separate textbook
/// loops. Shares nothing with the fused pass except the formula.
#[allow(clippy::too_many_arguments)]
fn reference(
    x: &[f64],
    targets: &[f64],
    weights: Option<&[f64]>,
    w: &[f64],
    (n, d): (usize, usize),
    fit_intercept: bool,
    sigma: f64,
    epsilon: f64,
) -> HuberEval {
    let d_aug = if fit_intercept { d + 1 } else { d };
    // The augmented design, materialized (the fused pass never does this).
    let mut xa = vec![0.0f64; n * d_aug];
    for r in 0..n {
        xa[r * d_aug..r * d_aug + d].copy_from_slice(&x[r * d..(r + 1) * d]);
        if fit_intercept {
            xa[r * d_aug + d] = 1.0;
        }
    }
    let mut margins = vec![0.0f64; n];
    for r in 0..n {
        let mut m = 0.0;
        for j in 0..d_aug {
            m += xa[r * d_aug + j] * w[j];
        }
        margins[r] = m;
    }
    let mut sq_sum = 0.0;
    let mut out_abs_sum = 0.0;
    let mut out_sw_sum = 0.0;
    let mut n_outliers = 0usize;
    let mut g = vec![0.0f64; n];
    for r in 0..n {
        let s = weights.map(|v| v[r]).unwrap_or(1.0);
        let res = targets[r] - margins[r];
        let a = res.abs();
        if a > epsilon * sigma {
            out_abs_sum += s * a;
            out_sw_sum += s;
            n_outliers += 1;
            g[r] = -2.0 * epsilon * s * if res < 0.0 { -1.0 } else { 1.0 };
        } else {
            sq_sum += s * res * res;
            g[r] = -2.0 * s * res / sigma;
        }
    }
    let mut xtg = vec![0.0f64; d_aug];
    for (j, slot) in xtg.iter_mut().enumerate() {
        let mut acc = 0.0;
        for r in 0..n {
            acc += xa[r * d_aug + j] * g[r];
        }
        *slot = acc;
    }
    HuberEval {
        sq_sum,
        out_abs_sum,
        out_sw_sum,
        n_outliers,
        xtg,
    }
}

/// The full objective the estimator assembles, recomputed here from a
/// [`HuberEval`] so the finite-difference check has something scalar to
/// differentiate.
fn total_loss(ev: &HuberEval, w: &[f64], sigma: f64, epsilon: f64, alpha: f64, n_features: usize, sw_total: f64) -> f64 {
    let squared_loss = ev.sq_sum / sigma;
    let outlier_loss = 2.0 * epsilon * ev.out_abs_sum - sigma * ev.out_sw_sum * epsilon * epsilon;
    let penalty: f64 = w[..n_features].iter().map(|v| v * v).sum();
    sw_total * sigma + squared_loss + outlier_loss + alpha * penalty
}

fn f_to<F: Pod>(v: f64) -> F {
    f64_to_host::<F>(v)
}

fn assert_close(got: f64, expected: f64, tol: f64, what: &str) {
    let err = (got - expected).abs();
    assert!(
        err <= tol + tol * expected.abs(),
        "{what}: got={got:e} expected={expected:e} err={err:e} (tol={tol:e})"
    );
}

/// One evaluation, both arms, compared against the naive reference.
#[allow(clippy::too_many_arguments)]
fn check_eval<F>(
    seed: u64,
    (n, d): (usize, usize),
    fit_intercept: bool,
    weighted: bool,
    sigma: f64,
    epsilon: f64,
    tol: f64,
    label: &str,
) where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x64 = uniform_pm1(seed, n * d);
    let targets: Vec<f64> = uniform_pm1(seed ^ 0xABCD, n)
        .iter()
        .map(|v| v * 3.0)
        .collect();
    // Strictly positive, non-uniform.
    let weights: Option<Vec<f64>> = weighted.then(|| {
        uniform_pm1(seed ^ 0x1234, n)
            .iter()
            .map(|v| 1.5 + v)
            .collect()
    });
    let d_aug = if fit_intercept { d + 1 } else { d };
    let w: Vec<f64> = uniform_pm1(seed ^ 0x55AA, d_aug);

    // The design as the caller actually holds it, in the storage dtype.
    let x_host: Vec<F> = x64.iter().map(|&v| f_to::<F>(v)).collect();
    // Reference against the ROUND-TRIPPED design, so an f32 case is not being
    // asked to reproduce f64 inputs it never saw.
    let x_rt: Vec<f64> = x_host.iter().map(|&v| host_f64(v)).collect();
    let expected = reference(
        &x_rt,
        &targets,
        weights.as_deref(),
        &w,
        (n, d),
        fit_intercept,
        sigma,
        epsilon,
    );

    for host_ingress in [true, false] {
        let xd: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
        let design = if host_ingress {
            HuberDesign::Host(x_host.as_slice())
        } else {
            HuberDesign::Device(&xd)
        };
        let obj = HuberObjective::<F>::new(
            &mut pool,
            design,
            (n, d),
            targets.clone(),
            weights.clone(),
            fit_intercept,
        )
        .expect("HuberObjective::new rejected a valid geometry");
        assert_eq!(obj.d_aug(), d_aug, "{label}: d_aug");
        let got = obj.eval(&mut pool, &w, sigma, epsilon).expect("eval");

        let what = |s: &str| format!("{label}/host_ingress={host_ingress}::{s}");
        assert_close(got.sq_sum, expected.sq_sum, tol, &what("sq_sum"));
        assert_close(
            got.out_abs_sum,
            expected.out_abs_sum,
            tol,
            &what("out_abs_sum"),
        );
        assert_close(
            got.out_sw_sum,
            expected.out_sw_sum,
            tol,
            &what("out_sw_sum"),
        );
        assert_eq!(
            got.n_outliers,
            expected.n_outliers,
            "{}: outlier COUNT differs",
            what("n_outliers")
        );
        assert_eq!(got.xtg.len(), d_aug, "{}: xtg length", what("xtg"));
        for (j, (&g, &e)) in got.xtg.iter().zip(expected.xtg.iter()).enumerate() {
            assert_close(g, e, tol, &what(&format!("xtg[{j}]")));
        }

        // The outlier mask is the same predicate, so it must agree with the
        // count the reduction produced.
        let mask = obj
            .outlier_mask(&mut pool, &w, sigma, epsilon)
            .expect("outlier_mask");
        assert_eq!(mask.len(), n, "{}: mask length", what("outlier_mask"));
        assert_eq!(
            mask.iter().filter(|&&m| m).count(),
            expected.n_outliers,
            "{}: mask disagrees with the reduction's own count",
            what("outlier_mask")
        );

        obj.release_into(&mut pool);
        xd.release_into(&mut pool);
    }
}

fn host_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("huber_objective is f32/f64 only"),
    }
}

#[test]
fn matches_naive_reference_f32() {
    capability::log_oracle_dtype(
        FloatKind::F32,
        capability::active_backend_name(),
        "huber_objective",
    );
    // Intercept on/off x weighted/unweighted, at a σ that puts a real mix of
    // samples on both sides of the ε·σ threshold.
    check_eval::<f32>(11, (57, 7), true, false, 0.8, 1.35, 1e-4, "f32 int/unweighted");
    check_eval::<f32>(12, (57, 7), false, false, 0.8, 1.35, 1e-4, "f32 noint/unweighted");
    check_eval::<f32>(13, (57, 7), true, true, 0.8, 1.35, 1e-4, "f32 int/weighted");
    check_eval::<f32>(14, (57, 7), false, true, 0.8, 1.35, 1e-4, "f32 noint/weighted");
}

#[test]
fn matches_naive_reference_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_eval::<f64>(11, (57, 7), true, false, 0.8, 1.35, 1e-9, "f64 int/unweighted");
    check_eval::<f64>(12, (57, 7), false, false, 0.8, 1.35, 1e-9, "f64 noint/unweighted");
    check_eval::<f64>(13, (57, 7), true, true, 0.8, 1.35, 1e-9, "f64 int/weighted");
    check_eval::<f64>(14, (57, 7), false, true, 0.8, 1.35, 1e-9, "f64 noint/weighted");
    // `d_aug` boundary: d = DOT_LANES exactly, so the dot product's
    // `chunks_exact` remainder is empty on the unaugmented block and non-empty
    // on nothing — the off-by-one the hoisted synthetic column could hide.
    check_eval::<f64>(15, (33, 8), true, false, 0.8, 1.35, 1e-9, "f64 d=8 lanes");
    check_eval::<f64>(16, (33, 9), true, true, 0.8, 1.35, 1e-9, "f64 d=9 remainder");
}

/// The extreme σ values, where every sample falls on ONE side of the threshold.
///
/// `σ → large` makes everything an inlier (the objective is plain weighted least
/// squares over `σ`), `σ → small` makes everything an outlier (weighted
/// least-absolute-deviations). Both branches of the fused loop's `select` have
/// to be right in isolation, not just on a mixed input.
#[test]
fn all_inlier_and_all_outlier_limits() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_eval::<f64>(21, (40, 5), true, false, 1e6, 1.35, 1e-9, "all-inlier");
    check_eval::<f64>(22, (40, 5), true, true, 1e-6, 1.35, 1e-9, "all-outlier");
}

/// Above `HUBER_ELEMS_PER_UNIT` (`1 << 14`) the pass fans out across the worker
/// pool and every partial is reduced afterwards. A design of `4000 × 16` is
/// 64 000 elements, so several workers engage and the per-worker chunking of the
/// design, the targets AND the weights all have to line up.
#[test]
fn multi_worker_split_reduces_correctly() {
    if capability::skip_f64_with_log() {
        return;
    }
    check_eval::<f64>(31, (4000, 16), true, false, 0.9, 1.35, 1e-9, "pool/unweighted");
    check_eval::<f64>(32, (4000, 16), true, true, 0.9, 1.35, 1e-9, "pool/weighted");
    // A row count that does NOT divide evenly by the worker count, so the last
    // chunk is short.
    check_eval::<f64>(33, (4001, 16), false, true, 0.9, 1.35, 1e-9, "pool/ragged");
}

/// The gradient the estimator assembles — including `∂L/∂σ` — against CENTRAL
/// DIFFERENCES of the objective value.
///
/// This is the check that shares no code with what it validates: it differences
/// the scalar loss only, so a sign error or a missing factor of two anywhere in
/// the gradient assembly shows up as a mismatch even if the fused pass and the
/// naive reference happen to agree with each other. `∂L/∂σ` is the entry that
/// most needs it — it is the difference of two `O(n)` quantities that nearly
/// cancel, and no other test in this file would see it be wrong.
#[test]
fn gradient_matches_central_differences() {
    if capability::skip_f64_with_log() {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (n, d) = (63, 5);
    let alpha = 0.37;
    let epsilon = 1.35;
    let x: Vec<f64> = uniform_pm1(41, n * d);
    let targets: Vec<f64> = uniform_pm1(42, n).iter().map(|v| v * 3.0).collect();
    let weights: Vec<f64> = uniform_pm1(43, n).iter().map(|v| 1.5 + v).collect();
    let sw_total: f64 = weights.iter().sum();

    for fit_intercept in [true, false] {
        let d_aug = if fit_intercept { d + 1 } else { d };
        let obj = HuberObjective::<f64>::new(
            &mut pool,
            HuberDesign::Host(&x),
            (n, d),
            targets.clone(),
            Some(weights.clone()),
            fit_intercept,
        )
        .expect("HuberObjective::new");

        // A base point away from any sample's ε·σ threshold, so the piecewise
        // objective is smooth in the whole finite-difference stencil (it is C¹
        // everywhere, but differencing ACROSS a knot costs an order of accuracy
        // and would make the check ambiguous).
        let mut p: Vec<f64> = uniform_pm1(44, d_aug + 1).iter().map(|v| v * 0.4).collect();
        p[d_aug] = 0.9;

        let loss_at = |obj: &HuberObjective<'_, f64>,
                       pool: &mut BufferPool<ActiveRuntime>,
                       p: &[f64]| -> f64 {
            let ev = obj.eval(pool, &p[..d_aug], p[d_aug], epsilon).expect("eval");
            total_loss(&ev, &p[..d_aug], p[d_aug], epsilon, alpha, d, sw_total)
        };

        // The analytic gradient, assembled exactly as `huber.rs` does.
        let ev = obj.eval(&mut pool, &p[..d_aug], p[d_aug], epsilon).expect("eval");
        let mut analytic = ev.xtg.clone();
        for (g, &wv) in analytic[..d].iter_mut().zip(&p[..d]) {
            *g += 2.0 * alpha * wv;
        }
        analytic.push(sw_total - ev.out_sw_sum * epsilon * epsilon - (ev.sq_sum / p[d_aug]) / p[d_aug]);

        for j in 0..=d_aug {
            let h = 1e-6 * p[j].abs().max(1.0);
            let mut hi = p.clone();
            let mut lo = p.clone();
            hi[j] += h;
            lo[j] -= h;
            let numeric = (loss_at(&obj, &mut pool, &hi) - loss_at(&obj, &mut pool, &lo)) / (2.0 * h);
            let scale = numeric.abs().max(analytic[j].abs()).max(1.0);
            assert!(
                (numeric - analytic[j]).abs() <= 1e-5 * scale,
                "fit_intercept={fit_intercept} grad[{j}] ({}): analytic={:e} central-difference={:e}",
                if j == d_aug { "sigma" } else if j == d { "intercept" } else { "coef" },
                analytic[j],
                numeric
            );
        }
        obj.release_into(&mut pool);
    }
}

/// Geometry is validated BEFORE anything is allocated (ASVS V5).
#[test]
fn rejects_invalid_geometry() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x = vec![0.0f32; 12];

    // n·d disagrees with the slab length.
    assert!(matches!(
        HuberObjective::<f32>::new(&mut pool, HuberDesign::Host(&x), (5, 3), vec![0.0; 5], None, true),
        Err(PrimError::ShapeMismatch { operand: "x", .. })
    ));
    // Zero rows / zero features.
    assert!(matches!(
        HuberObjective::<f32>::new(&mut pool, HuberDesign::Host(&x), (0, 3), vec![], None, true),
        Err(PrimError::ShapeMismatch { operand: "x", .. })
    ));
    // targets length.
    assert!(matches!(
        HuberObjective::<f32>::new(&mut pool, HuberDesign::Host(&x), (4, 3), vec![0.0; 3], None, true),
        Err(PrimError::ShapeMismatch {
            operand: "targets",
            ..
        })
    ));
    // sample_weight length.
    assert!(matches!(
        HuberObjective::<f32>::new(
            &mut pool,
            HuberDesign::Host(&x),
            (4, 3),
            vec![0.0; 4],
            Some(vec![1.0; 3]),
            true
        ),
        Err(PrimError::ShapeMismatch {
            operand: "sample_weight",
            ..
        })
    ));
    // `w` of the wrong augmented length is rejected at eval, not silently read.
    let obj = HuberObjective::<f32>::new(
        &mut pool,
        HuberDesign::Host(&x),
        (4, 3),
        vec![0.0; 4],
        None,
        true,
    )
    .expect("valid geometry");
    assert!(matches!(
        obj.eval(&mut pool, &[0.0; 3], 1.0, 1.35),
        Err(PrimError::DimMismatch { dim: "d_aug", .. })
    ));
    assert!(matches!(
        obj.outlier_mask(&mut pool, &[0.0; 3], 1.0, 1.35),
        Err(PrimError::DimMismatch { dim: "d_aug", .. })
    ));
    obj.release_into(&mut pool);
}
