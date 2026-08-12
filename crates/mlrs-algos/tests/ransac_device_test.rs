//! `RANSACRegressor`'s DEVICE scan arm (RANSAC-02) — batching, rewind, and
//! arm agreement.
//!
//! The device arm exists to amortize a launch and a host stall across a BATCH of
//! trials, which is only sound because the trials in a batch are mutually
//! independent and the bookkeeping is replayed over them in order afterwards
//! (`mlrs_algos::linear::ransac` module docs). Three things have to hold for
//! that to be a pure performance change, and this file is those three:
//!
//! 1. **The two arms agree.** Same counters, same consensus, same coefficients
//!    — checked against the committed oracle fixture so "agree" means "agree
//!    with sklearn", not merely "agree with each other".
//! 2. **The batch width does not change the fit.** Widths 1, 3, 8 and 13 (a
//!    width that does not divide any of the trial counts, deliberately) must all
//!    produce the identical fitted state.
//! 3. **The draw stream is rewound.** A stop rule firing mid-batch must leave
//!    the caller's `RandomState` exactly where an unbatched loop would have —
//!    otherwise every subsequent draw the caller makes is wrong, which is a
//!    silent failure no fitted attribute would show.
//!
//! The arm is selected through `MLRS_RANSAC_ENGINE`, scoped per test thread via
//! [`abflag`](mlrs_backend::abflag) rather than `std::env::set_var`
//! ([[mlrs-abflag-test-knobs]]). On a `--features cpu` build this runs the same
//! kernels through `cubecl-cpu`, which is a legal `device="gpu"` request
//! (`mlrs_backend::device` module docs) and is what lets these hold on a machine
//! with no GPU at all — the [[mlrs-gaussian-mixture-cuda-device]] technique.
//!
//! Every test SKIPS with a reason if the arm did not actually engage
//! (`device_arm() != "gpu"`), because a vacuous pass here would be worse than a
//! skip: it would read as "the device arm agrees" when nothing device ran
//! ([[mlrs-bench-verify-knob-is-live]]).

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::ransac::{RansacDriver, RansacRegressor};
use mlrs_algos::model_selection::rng::NumpyRandomState;
use mlrs_algos::typestate::Fitted;
use mlrs_backend::abflag;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

const N_SAMPLES: usize = 300;
const N_FEATURES: usize = 5;
const SEED: u32 = 42;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn oracle() -> OracleCase {
    load_npz(&fixture("ransac_f32_seed42.npz")).expect("the committed RANSAC fixture")
}

fn as_f32(case: &OracleCase, name: &str) -> Vec<f32> {
    case.expect_f64(name).iter().map(|&v| v as f32).collect()
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

/// One fit, with the arm and the batch width forced, returning the fitted
/// estimator and the state the draw stream was left in.
fn fit_with(
    engine: &str,
    batch: Option<&str>,
    x: &[f32],
    y: &[f32],
    stop_n_inliers: f64,
) -> (RansacRegressor<f32, Fitted>, [u32; 624], usize) {
    let _engine = abflag::force("MLRS_RANSAC_ENGINE", engine);
    let _batch = batch.map(|b| abflag::force("MLRS_RANSAC_BATCH", b));
    let est = RansacRegressor::<f32>::builder()
        .stop_n_inliers(stop_n_inliers)
        .build::<f32>()
        .expect("sklearn's defaults are in range");
    let mut rng = NumpyRandomState::from_seed(SEED);
    let fitted = est
        .fit_from_host_slice(
            &mut pool(),
            x,
            y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut rng,
            &RansacDriver::default(),
        )
        .expect("the fixture design has a consensus set");
    (fitted, *rng.key(), rng.pos())
}

/// Everything a fit is allowed to differ in — nothing.
fn assert_same_fit<F: Float + CubeElement + Pod>(
    a: &RansacRegressor<F, Fitted>,
    b: &RansacRegressor<F, Fitted>,
    what: &str,
) {
    assert_eq!(a.n_trials(), b.n_trials(), "{what}: n_trials_");
    assert_eq!(
        a.n_skips_no_inliers(),
        b.n_skips_no_inliers(),
        "{what}: n_skips_no_inliers_"
    );
    assert_eq!(
        a.n_skips_invalid_data(),
        b.n_skips_invalid_data(),
        "{what}: n_skips_invalid_data_"
    );
    assert_eq!(
        a.n_skips_invalid_model(),
        b.n_skips_invalid_model(),
        "{what}: n_skips_invalid_model_"
    );
    assert_eq!(a.inlier_mask(), b.inlier_mask(), "{what}: inlier_mask_");
    for (i, (&l, &r)) in a.coef().iter().zip(b.coef()).enumerate() {
        assert!(
            (l - r).abs() <= 1e-12 * l.abs().max(1.0),
            "{what}: coef_[{i}] {l} != {r}"
        );
    }
}

/// The two arms produce the same fit, and the device one really ran.
#[test]
fn the_device_arm_agrees_with_the_host_arm() {
    let case = oracle();
    let (x, y) = (as_f32(&case, "X"), as_f32(&case, "y"));

    let (host, hk, hp) = fit_with("host", None, &x, &y, f64::INFINITY);
    let (dev, dk, dp) = fit_with("device", None, &x, &y, f64::INFINITY);
    if dev.device_arm() != "gpu" {
        eprintln!("SKIP: the device scan is unavailable on this backend");
        return;
    }
    assert!(
        dev.batch_width() > 1,
        "the device arm must batch; got a width of {}",
        dev.batch_width()
    );
    assert_eq!(host.device_arm(), "cpu", "the forced host arm must be host");
    assert_same_fit(&host, &dev, "host vs device");
    // The draw stream must also end in the same place — the device arm draws
    // SPECULATIVELY and rewinds, so this is the property that says the rewind
    // put every surplus word back.
    assert_eq!(
        (hk, hp),
        (dk, dp),
        "the draw stream ended in a different state"
    );
}

/// The batch width is a performance knob and nothing else.
#[test]
fn batching_does_not_change_the_fit() {
    let case = oracle();
    let (x, y) = (as_f32(&case, "X"), as_f32(&case, "y"));

    let (one, k1, p1) = fit_with("device", Some("1"), &x, &y, f64::INFINITY);
    if one.device_arm() != "gpu" {
        eprintln!("SKIP: the device scan is unavailable on this backend");
        return;
    }
    // 13 is deliberately awkward: it divides neither `max_trials` nor the trial
    // count any of these fits actually reaches, so the last batch is always a
    // partial one and the rewind path is always exercised.
    for width in ["3", "8", "13"] {
        let (other, k, p) = fit_with("device", Some(width), &x, &y, f64::INFINITY);
        assert_same_fit(&one, &other, &format!("batch width {width}"));
        assert_eq!(
            (k1, p1),
            (k, p),
            "batch width {width}: the draw stream ended in a different state"
        );
    }
}

/// A stop rule firing mid-batch rewinds the draw stream to exactly where the
/// consumed trials left it.
///
/// `stop_n_inliers` is the lever because it fires on the FIRST trial that
/// reaches it, which — at a batch width of eight — is almost never the last
/// trial of its batch. The unbatched host arm is the reference: it cannot
/// speculate, so its final stream state is by construction the right answer.
#[test]
fn a_mid_batch_stop_rewinds_the_draw_stream() {
    let case = oracle();
    let (x, y) = (as_f32(&case, "X"), as_f32(&case, "y"));

    for stop in [60.0, 120.0, 200.0] {
        let (host, hk, hp) = fit_with("host", None, &x, &y, stop);
        let (dev, dk, dp) = fit_with("device", Some("8"), &x, &y, stop);
        if dev.device_arm() != "gpu" {
            eprintln!("SKIP: the device scan is unavailable on this backend");
            return;
        }
        assert_same_fit(&host, &dev, &format!("stop_n_inliers = {stop}"));
        assert_eq!(
            (hk, hp),
            (dk, dp),
            "stop_n_inliers = {stop}: the surplus draws were not rewound"
        );
    }
}
