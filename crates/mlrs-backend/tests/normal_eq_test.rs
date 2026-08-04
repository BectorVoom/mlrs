//! Device normal-equations formation (`prims::normal_eq::centered_gram_xty_device`)
//! oracle validation, with the WEIGHTED arm as the subject.
//!
//! ## Why this suite exists
//! The weighted arm reaches the `col_sums_blocked`/`col_sums_reduce` kernel pair
//! through its own private launch site (`column_sums_scaled`), which is the
//! SECOND caller of that pair — `prims::gram::column_means` is the first. Those
//! kernels gained a `weighted` arm and a multi-target `k` on one branch while
//! this second caller was added on another; the two met at a textually clean
//! merge, and `main` stopped compiling on every backend.
//!
//! A stale launch list is normally caught by a compiler, so the interesting
//! question is what happens once it compiles again — a `weighted`/`k`/`inv`
//! argument that is merely WRONG is silent. Nothing gated that: the end-to-end
//! test on this path (`bayesian_ridge_test::device_gram_agrees`) is VACUOUS
//! under `--features cpu`, because `device_gram_applicable` returns `false` on
//! the cpu backend before it ever consults the A/B flag, so both of its arms
//! are the host sweep there.
//!
//! This suite calls `centered_gram_xty_device` DIRECTLY rather than through that
//! predicate. The predicate is a throughput decision — the device formation is
//! correct on cpu, merely slow — so bypassing it is what makes the launch site
//! actually EXECUTE in cpu CI rather than only type-check.
//!
//! The reference is a deliberately naive f64 triple loop, not a re-derivation of
//! the blocked reduction, so a bug in the row blocking, the partial fold or the
//! launch arguments cannot be reproduced by the oracle.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `#[cfg(test)] mod tests`
//! in `src/`.

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::normal_eq::centered_gram_xty_device;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{assert_slice_close, F64_TOL};

/// Deterministic, reproducible pseudo-random design (splitmix64), so a failure
/// is a fixed case rather than one that depends on the run.
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

/// `(x, y, w)` for one case.
///
/// Column means are deliberately far from zero (each column offset by
/// `1 + j`), so a dropped, mis-indexed or mis-scaled mean is a LARGE error
/// rather than a rounding-scale one — the failure mode a stale launch argument
/// produces.
///
/// The weights span an order of magnitude and are NOT a permutation of a
/// constant, so `Σwᵣxᵣ/Σw` is far from the unweighted mean: a `weighted`
/// argument that silently flipped, or an `inv` that reverted to `1/n`, moves
/// the answer well outside `F64_TOL`.
fn make_case(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let x: Vec<f64> = (0..n * d)
        .map(|k| uniform_pm1(&mut s) + 1.0 + (k % d) as f64)
        .collect();
    let y: Vec<f64> = (0..n).map(|_| uniform_pm1(&mut s) + 4.0).collect();
    let w: Vec<f64> = (0..n)
        .map(|_| 0.25 + 2.0 * (uniform_pm1(&mut s) + 1.0))
        .collect();
    (x, y, w)
}

/// The naive reference: weighted means, then an explicit `√w`-scaled centered
/// design, then a plain triple loop over it (`gram_host_test`'s oracle, which
/// is independent of both implementations under comparison).
#[allow(clippy::type_complexity)]
fn reference(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    sw: Option<&[f64]>,
    fit_intercept: bool,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let mut xm = vec![0.0f64; d];
    let mut ym = 0.0f64;
    if fit_intercept {
        let wsum: f64 = match sw {
            Some(w) => w.iter().sum(),
            None => n as f64,
        };
        for i in 0..n {
            let w = sw.map(|w| w[i]).unwrap_or(1.0);
            for (a, m) in xm.iter_mut().enumerate() {
                *m += w * x[i * d + a];
            }
            ym += w * y[i];
        }
        for m in xm.iter_mut() {
            *m /= wsum;
        }
        ym /= wsum;
    }

    let mut gram = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];
    for i in 0..n {
        let scale = sw.map(|w| w[i].sqrt()).unwrap_or(1.0);
        let row: Vec<f64> = (0..d).map(|a| (x[i * d + a] - xm[a]) * scale).collect();
        let yc = (y[i] - ym) * scale;
        for a in 0..d {
            xty[a] += row[a] * yc;
            for b in 0..d {
                gram[a * d + b] += row[a] * row[b];
            }
        }
    }
    (xm, ym, gram, xty)
}

/// Shapes chosen around `row_blocking`'s `nb = ⌈n/256⌉` split, because the
/// number of row BLOCKS is what the two-kernel fold is sensitive to: `nb = 1`
/// exercises the degenerate fold, `nb = 2` the first real one, and `nb = 3` a
/// fold deep enough that a partial written to the wrong `b · d + c` slot cannot
/// coincidentally land on its own block.
///
/// They are deliberately TALL AND NARROW — `n` buys the block count, `d` is
/// held to single digits — because the cpu runtime spawns one OS thread per
/// unit, so the elementwise passes this arm runs (`row_scale_center` over
/// `n · d`) cost roughly linearly in `n · d` there: measured ~9 ms per element,
/// i.e. a single `(513, 8)` case is ~38 s. Total `n · d` across the suite is
/// ~2.4 k, which keeps the whole suite near a second there.
///
/// What that trades away is the `d > CUBE_DIM_X` (64) column walk, where a unit
/// folds more than one column. That walk is shared verbatim with
/// `prims::gram::column_means` and is covered at full width by `gram_test`;
/// what is unique to THIS site, and what these shapes gate, is the launch
/// argument list and the block fold.
const SHAPES: &[(usize, usize)] = &[(64, 4), (300, 3), (600, 2)];

/// The weighted device arm vs the naive host reference — the gate on
/// `column_sums_scaled`'s launch site.
///
/// Both the means and the Gram are asserted. The means alone are what
/// `column_sums_scaled` produces, but a mean that is wrong propagates into the
/// centered Gram quadratically, so comparing both distinguishes "the mean pass
/// is wrong" from "the Gram pass is wrong" when this fails.
#[test]
fn centered_gram_xty_device_weighted_matches_ref_f64() {
    let backend = capability::active_backend_name();
    if capability::skip_f64_with_log() {
        println!("normal_eq weighted f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y, w) = make_case(n, d, 42 + i as u64);
        let xd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
        let yd: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

        let (xm, ym, gram, xty) =
            centered_gram_xty_device::<f64>(&mut pool, &xd, &yd, n, d, Some(&w), true)
                .expect("weighted device assembly must succeed");
        let (exp_xm, exp_ym, exp_gram, exp_xty) = reference(&x, &y, n, d, Some(&w), true);

        assert_slice_close(&xm, &exp_xm, &F64_TOL);
        assert_slice_close(&[ym], &[exp_ym], &F64_TOL);
        assert_slice_close(&gram, &exp_gram, &F64_TOL);
        assert_slice_close(&xty, &exp_xty, &F64_TOL);

        xd.release_into(&mut pool);
        yd.release_into(&mut pool);
    }
}

/// The fixture must actually DISCRIMINATE weighted means from unweighted ones.
///
/// This is the one property a stale launch site can break while still producing
/// a plausible-looking, finite, correctly-shaped answer: `weighted = 0` with an
/// `inv` that reverted to `1/n` is exactly the unweighted mean, and it would
/// satisfy every shape and finiteness check in the suite. So the gap between
/// the two means is asserted to be large before the test above is trusted to
/// mean anything.
///
/// Both means come from the host `reference`, not from a second device call:
/// the unweighted device arm routes through the fused
/// `column_means` + `gram_xty_centered` composition, which is a different code
/// path (already gated by `gram_test`) and is ~75× more expensive than the
/// whole of the rest of this suite under the cpu runtime. What is being checked
/// here is a property of the FIXTURE, and the fixture is host data.
#[test]
fn fixture_weights_discriminate_the_means() {
    for (i, &(n, d)) in SHAPES.iter().enumerate() {
        let (x, y, w) = make_case(n, d, 42 + i as u64);
        let (w_xm, w_ym, _, _) = reference(&x, &y, n, d, Some(&w), true);
        let (u_xm, u_ym, _, _) = reference(&x, &y, n, d, None, true);

        let gap = (w_ym - u_ym).abs().max(
            w_xm.iter()
                .zip(&u_xm)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max),
        );
        assert!(
            gap > 1e-3,
            "shape ({n}, {d}) does not discriminate weighted from unweighted means \
             (gap={gap:e}); the weighted-path assertions would be vacuous"
        );
    }
}
