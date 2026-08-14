//! `prims::ridge_gcv` — the device `RidgeCV` engine's two phases, validated
//! against naive `f64` references.
//!
//! ## Why this suite calls the prim DIRECTLY
//! `gcv_device_possible` is `false` on the cpu backend (its `fused_centering_
//! available` gate is, because `gram_path` is a hard-wired `Gemm` there) and
//! `false` on wgpu (no `f64` device kernels). So the END-TO-END gate
//! — `ridge_cv_device_test::arms_agree` — is VACUOUS on both of the backends
//! this repo gates CI on: it skips, and the launch sites here are only
//! type-checked. That is the exact hole `normal_eq_test.rs` was written to
//! close for `BayesianRidge`, after a stale `col_sums_*` argument list slipped
//! through a clean merge.
//!
//! Bypassing the predicate is legitimate because it is a THROUGHPUT decision:
//! the kernels are correct on the cpu runtime, merely slow. What the bypass buys
//! is that a wrong `weighted`/`n_y`/`nblocks` launch argument — the failure mode
//! a compiler cannot see — fails in cpu CI.
//!
//! ## The reference is deliberately not the implementation
//! Both references are naive triple loops over the definition. The sweep
//! reference in particular re-derives `W = X̃·V` row by row with no tiling, no
//! blocking and no shared memory, so a bug in the row tile, the partial fold or
//! the `(i, a, t)` output indexing cannot be reproduced by the oracle.
//!
//! ## The operands are synthetic, and that is the point
//! `sweep` is handed a `V` / `g` / `gz` / `gzsw` that are NOT the fixture's
//! eigendecomposition. Nothing in the kernel requires them to be — it contracts
//! whatever it is given — so arbitrary operands are the stronger test: a
//! transposed `V` read or a `gz` indexed `[k·n_alphas + t]` instead of
//! `[(a·d + k)·n_y + t]` would still look plausible against a real
//! eigendecomposition's near-symmetric operands and cannot hide here. `V` is
//! scaled so the LOO denominator `1 − q` stays near 1: the identity being
//! divided by is a cancellation, and a reference that straddled zero would
//! measure the fixture rather than the kernel.
//!
//! Shapes are TINY on purpose. The cpu runtime spawns one OS thread per unit and
//! JITs at `-O0`, so the elementwise `center_scale` pass the weighted arm runs
//! costs roughly linearly in `n · d` there (`normal_eq_test`'s measured ~9 ms per
//! element). `n · d ≤ 200` per case keeps the suite near a second on cpu while
//! still crossing every branch: weighted / unweighted, intercept / no intercept,
//! single / multi target, scores / predictions.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::ridge_gcv::GcvDevice;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Deterministic pseudo-random source (splitmix64), so a failure is a fixed
/// case rather than one that depends on the run.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)`.
fn uniform_pm1(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// One fixture: `(x, y, w)`.
///
/// Column means are offset by `1 + j` and the target by `4`, so a dropped or
/// mis-indexed mean is a LARGE error rather than a rounding-scale one. The
/// weights span an order of magnitude and are not a permutation of a constant,
/// so the weighted means are far from the unweighted ones.
fn make_case(n: usize, d: usize, n_y: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..n * d)
        .map(|k| uniform_pm1(&mut s) + 1.0 + (k % d) as f64)
        .collect();
    let y: Vec<f64> = (0..n * n_y)
        .map(|k| uniform_pm1(&mut s) + 4.0 + (k % n_y) as f64)
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|_| 0.25 + 2.0 * (uniform_pm1(&mut s) + 1.0))
        .collect();
    (x, y, w)
}

/// `√w`, or ones.
fn sqrt_sw_of(w: Option<&[f64]>, n: usize) -> Vec<f64> {
    match w {
        Some(w) => w.iter().map(|v| v.sqrt()).collect(),
        None => vec![1.0; n],
    }
}

/// The naive reference for [`GcvDevice::normal_equations`].
#[allow(clippy::type_complexity)]
fn reference_normal_equations(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    n_y: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut xm = vec![0.0f64; d];
    let mut ym = vec![0.0f64; n_y];
    if fit_intercept {
        let wsum: f64 = match sw {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };
        for i in 0..n {
            let w = sw.map(|w| w[i]).unwrap_or(1.0);
            for (j, m) in xm.iter_mut().enumerate() {
                *m += w * x[i * d + j];
            }
            for (t, m) in ym.iter_mut().enumerate() {
                *m += w * y[i * n_y + t];
            }
        }
        for m in xm.iter_mut() {
            *m /= wsum;
        }
        for m in ym.iter_mut() {
            *m /= wsum;
        }
    }

    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d * n_y];
    let mut xtsw = vec![0.0f64; d];
    for i in 0..n {
        let s = sw.map(|w| w[i].sqrt()).unwrap_or(1.0);
        let row: Vec<f64> = (0..d).map(|j| (x[i * d + j] - xm[j]) * s).collect();
        for a in 0..d {
            for t in 0..n_y {
                xty[a * n_y + t] += row[a] * (y[i * n_y + t] - ym[t]) * s;
            }
            if fit_intercept {
                xtsw[a] += row[a] * s;
            }
            for b in 0..d {
                gram[a * d + b] += row[a] * row[b];
            }
        }
    }
    (xm, ym, gram, xty, xtsw)
}

/// The naive reference for [`GcvDevice::sweep`]: `W = X̃·V` row by row, then the
/// LOO identity, straight from `ridge_cv.rs`'s documented formulas.
#[allow(clippy::too_many_arguments)]
fn reference_sweep(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    n_y: usize,
    sqrt_sw: &[f64],
    sw_sum: f64,
    weighted: bool,
    fit_intercept: bool,
    x_offset: &[f64],
    y_offset: &[f64],
    v: &[f64],
    g: &[f64],
    gz: &[f64],
    gzsw: &[f64],
    n_alphas: usize,
    want_predictions: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut score_sums = vec![0.0f64; n_alphas * n_y];
    let mut values = vec![0.0f64; n * n_alphas * n_y];
    for i in 0..n {
        let s_i = sqrt_sw[i];
        let xt: Vec<f64> = (0..d).map(|j| s_i * (x[i * d + j] - x_offset[j])).collect();
        let w: Vec<f64> = (0..d)
            .map(|k| (0..d).map(|j| xt[j] * v[j * d + k]).sum())
            .collect();
        for a in 0..n_alphas {
            let mut q = 0.0f64;
            let mut rsw = 0.0f64;
            for k in 0..d {
                q += g[a * d + k] * w[k] * w[k];
                rsw += w[k] * gzsw[a * d + k];
            }
            let mut denom = 1.0 - q;
            if fit_intercept {
                denom -= (s_i - rsw) * s_i / sw_sum;
            }
            for t in 0..n_y {
                let tt: f64 = (0..d).map(|k| w[k] * gz[(a * d + k) * n_y + t]).sum();
                let yt = s_i * (y[i * n_y + t] - y_offset[t]);
                let looe = (yt - tt) / denom;
                if want_predictions {
                    let mut p = yt - looe;
                    if weighted {
                        p /= s_i;
                    }
                    p += y_offset[t];
                    values[(i * n_alphas + a) * n_y + t] = p;
                } else {
                    let sq = looe * looe;
                    score_sums[a * n_y + t] += sq;
                    values[(i * n_alphas + a) * n_y + t] = sq;
                }
            }
        }
    }
    (score_sums, values)
}

/// Synthetic `(v, g, gz, gzsw)` — see the module docs for why they are not the
/// fixture's eigendecomposition, and why `v` is scaled down.
fn make_operands(d: usize, n_y: usize, n_alphas: usize, seed: u64) -> [Vec<f64>; 4] {
    let mut s = seed;
    // 0.05 keeps `‖W‖` small enough that `q ≪ 1` and the LOO denominator stays
    // near 1 — the reference must not be dominated by a cancellation.
    let v: Vec<f64> = (0..d * d).map(|_| 0.05 * uniform_pm1(&mut s)).collect();
    let g: Vec<f64> = (0..n_alphas * d)
        .map(|_| 0.5 + 0.5 * (uniform_pm1(&mut s) + 1.0))
        .collect();
    let gz: Vec<f64> = (0..n_alphas * d * n_y)
        .map(|_| uniform_pm1(&mut s))
        .collect();
    let gzsw: Vec<f64> = (0..n_alphas * d)
        .map(|_| 0.1 * uniform_pm1(&mut s))
        .collect();
    [v, g, gz, gzsw]
}

/// Relative-or-absolute agreement, the shape every `f64` device/host comparison
/// in this crate uses (the two arms differ in summation ORDER, never in the
/// expression evaluated).
fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-9 * b.abs().max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "{label}[{i}]: got {a}, want {b} (tol {tol})"
        );
    }
}

/// One configuration, end to end through both prim phases.
fn run_case(n: usize, d: usize, n_y: usize, weighted: bool, fit_intercept: bool, seed: u64) {
    // The whole engine is `f64` by construction (module docs), so a backend
    // without it has nothing to check rather than something to check loosely.
    if capability::skip_f64_with_log() {
        println!(
            "ridge_gcv n={n} d={d} n_y={n_y}: SKIPPED (no f64 on {})",
            capability::active_backend_name()
        );
        return;
    }
    let (x, y, w) = make_case(n, d, n_y, seed);
    let sw: Option<&[f64]> = if weighted { Some(&w) } else { None };
    let sqrt_sw = sqrt_sw_of(sw, n);
    let sw_sum: f64 = sqrt_sw.iter().map(|s| s * s).sum();
    let n_alphas = 3usize;
    let [v, g, gz, gzsw] = make_operands(d, n_y, n_alphas, seed ^ 0xABCD);

    let mut pool = BufferPool::<ActiveRuntime>::new(runtime::active_client());
    let dev = GcvDevice::from_host::<f64>(&mut pool, &x, &y, n, d, n_y, &sqrt_sw, weighted)
        .expect("upload");

    let (xm, ym, gram, xty, xtsw) = dev
        .normal_equations(&mut pool, fit_intercept)
        .expect("normal equations");
    let (rxm, rym, rgram, rxty, rxtsw) =
        reference_normal_equations(&x, &y, n, d, n_y, sw, fit_intercept);
    let label = format!("n={n} d={d} n_y={n_y} weighted={weighted} fi={fit_intercept}");
    assert_close(&xm, &rxm, &format!("{label} x_offset"));
    assert_close(&ym, &rym, &format!("{label} y_offset"));
    assert_close(&gram, &rgram, &format!("{label} gram"));
    assert_close(&xty, &rxty, &format!("{label} xty"));
    assert_close(&xtsw, &rxtsw, &format!("{label} xtsw"));

    for want_predictions in [false, true] {
        let out = dev
            .sweep(
                &mut pool,
                &xm,
                &ym,
                &v,
                &g,
                &gz,
                &gzsw,
                n_alphas,
                sw_sum,
                fit_intercept,
                want_predictions,
                true,
            )
            .expect("sweep");
        let (rsums, rvals) = reference_sweep(
            &x,
            &y,
            n,
            d,
            n_y,
            &sqrt_sw,
            sw_sum,
            weighted,
            fit_intercept,
            &xm,
            &ym,
            &v,
            &g,
            &gz,
            &gzsw,
            n_alphas,
            want_predictions,
        );
        let tag = format!("{label} pred={want_predictions}");
        assert_close(&out.cv_values, &rvals, &format!("{tag} cv_values"));
        if want_predictions {
            assert!(
                out.score_sums.is_empty(),
                "{tag}: predictions must not also produce scores"
            );
        } else {
            assert_close(&out.score_sums, &rsums, &format!("{tag} score_sums"));
        }
    }

    dev.release_into(&mut pool);
}

#[test]
fn unweighted_with_intercept_matches_the_reference() {
    run_case(32, 5, 1, false, true, 0x5EED_0001);
}

#[test]
fn weighted_with_intercept_matches_the_reference() {
    run_case(32, 5, 1, true, true, 0x5EED_0002);
}

/// `fit_intercept = false` drops the `(√w − rsw)·√w / Σw` term from the LOO
/// denominator AND zeroes the means. Both halves are checked, because zeroing
/// only one of them still produces a plausible-looking fit.
#[test]
fn no_intercept_matches_the_reference() {
    run_case(32, 5, 1, false, false, 0x5EED_0003);
}

#[test]
fn weighted_without_intercept_matches_the_reference() {
    run_case(24, 4, 1, true, false, 0x5EED_0004);
}

/// Multi-target is where the `(a·d + k)·n_y + t` indexing can go wrong without
/// changing any length, so it is checked with weights on (the arm that also
/// re-scales the prediction).
#[test]
fn multi_target_matches_the_reference() {
    run_case(24, 4, 3, false, true, 0x5EED_0005);
}

#[test]
fn multi_target_weighted_matches_the_reference() {
    run_case(24, 4, 2, true, true, 0x5EED_0006);
}

/// More than one row BLOCK, which is what the per-block partial fold is
/// sensitive to: a partial written to the wrong `(block, alpha, target)` slot
/// cannot coincidentally land on its own block once there are three.
#[test]
fn several_row_blocks_fold_correctly() {
    run_case(40, 4, 1, false, true, 0x5EED_0007);
}

/// An `n` that is NOT a multiple of `GCV_ROW_TILE`, so the last tile stages
/// dead lanes. Those lanes' `W` is computed and must be discarded; a missing
/// `rc < live` guard shows up here as extra rows folded into the scores.
#[test]
fn a_partial_row_tile_discards_its_dead_lanes() {
    run_case(23, 4, 1, false, true, 0x5EED_0008);
    run_case(21, 3, 2, true, true, 0x5EED_0009);
}

/// `f32` ingress: the design is uploaded at the estimator's own width and
/// widened ON THE DEVICE, so the sweep is still `f64`. The band is looser only
/// because the INPUT is `f32` — the arithmetic after the widening is not.
#[test]
fn f32_ingress_widens_on_the_device() {
    if capability::skip_f64_with_log() {
        println!(
            "ridge_gcv f32 ingress: SKIPPED (no f64 on {})",
            capability::active_backend_name()
        );
        return;
    }
    let (n, d, n_y) = (32usize, 5usize, 1usize);
    let (x64, y64, _) = make_case(n, d, n_y, 0x5EED_000A);
    let x: Vec<f32> = x64.iter().map(|v| *v as f32).collect();
    let y: Vec<f32> = y64.iter().map(|v| *v as f32).collect();
    let xw: Vec<f64> = x.iter().map(|v| *v as f64).collect();
    let yw: Vec<f64> = y.iter().map(|v| *v as f64).collect();
    let sqrt_sw = vec![1.0f64; n];

    let mut pool = BufferPool::<ActiveRuntime>::new(runtime::active_client());
    let dev =
        GcvDevice::from_host::<f32>(&mut pool, &x, &y, n, d, n_y, &sqrt_sw, false).expect("upload");
    let (xm, ym, gram, xty, xtsw) = dev.normal_equations(&mut pool, true).expect("ne");
    // The reference consumes the f32 values EXACTLY as widened, so any
    // disagreement is the device path's, not the narrowing's.
    let (rxm, rym, rgram, rxty, rxtsw) =
        reference_normal_equations(&xw, &yw, n, d, n_y, None, true);
    assert_close(&xm, &rxm, "f32 x_offset");
    assert_close(&ym, &rym, "f32 y_offset");
    assert_close(&gram, &rgram, "f32 gram");
    assert_close(&xty, &rxty, "f32 xty");
    assert_close(&xtsw, &rxtsw, "f32 xtsw");
    dev.release_into(&mut pool);
}
