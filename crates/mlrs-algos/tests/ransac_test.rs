//! `RANSACRegressor` (RANSAC-01) FULL sklearn parameter-surface oracle tests.
//!
//! Gates every `sklearn.linear_model.RANSACRegressor` parameter against the
//! committed `ransac_{f32,f64}_seed42` fixture (`scripts/gen_oracle.py`
//! ::`gen_ransac`).
//!
//! | parameter | how it is gated |
//! |---|---|
//! | `loss` | **the string-valued one** — both `"absolute_error"` and `"squared_error"` as value cases, plus [`loss_strings_round_trip_and_reject_the_unknown`] on the builder's `TryFrom`-style parse |
//! | `min_samples` | all THREE forms: `None` (→ `n_features + 1`), an absolute `20` / `1`, and the fraction `0.3` (→ `ceil(0.3·n)`), with the RESOLVED value gated per case |
//! | `residual_threshold` | the `None` default (→ the target MAD, gated as a number) plus a tight `0.4` and a loose `8.0` |
//! | `max_trials` | `3` (truncation) and `500` (a budget the dynamic rule never spends) |
//! | `stop_probability` | `0.1` / `0.999999` / `1.0`, each landing on a different `n_trials_` |
//! | `stop_n_inliers` | an early exit at `200` inliers |
//! | `stop_score` | an early exit at `R² ≥ 0.5` |
//! | `max_skips` | a case that BLOWS the budget and still finds a consensus — sklearn's `ConvergenceWarning` branch, surfaced here as `exceeded_max_skips()` |
//! | `estimator` | the base `LinearRegression`'s `fit_intercept=False` |
//! | `is_data_valid` / `is_model_valid` | predicates that really reject (the fixture asserts the skip counters moved), plus [`a_callback_can_abort_the_fit`] |
//! | `random_state` | implicitly by EVERY case — see below |
//! | `sample_weight` | four weighted cases crossed with `min_samples` / `loss` / `fit_intercept` |
//!
//! ## Why this suite gates the TRAJECTORY and not just the answer
//! mlrs reproduces numpy's MT19937 exactly
//! ([`NumpyRandomState`](mlrs_algos::model_selection::rng::NumpyRandomState)), so
//! for an integer `random_state` both libraries visit the same sub-samples in
//! the same order. Every case therefore asserts `n_trials_` and all three
//! `n_skips_*` counters for EXACT equality, and the full `inlier_mask_` too —
//! which is a far stronger statement than agreeing on the final coefficients,
//! and the one that catches a stopping rule or a skip counter that drifted. A
//! `coef_` comparison alone would pass for an implementation that took a
//! different route to a similar consensus.
//!
//! ## The one thing that is NOT bit-exact, and how the fixture handles it
//! Residuals come from `X·coef` — a BLAS `gemv` in numpy, a lane-split host dot
//! product here — so a row sitting *exactly* on `residual_threshold` could land
//! on the other side of it. `gen_ransac` MEASURES how close the nearest row gets
//! (relative to the residual scale) and ships it as `margin_<name>`; this suite
//! gates `inlier_mask_` for exact equality only where that margin is
//! comfortable ([`MASK_EXACT_MARGIN`]) and falls back to the inlier COUNT
//! elsewhere. One case — `max_skips`, whose `residual_threshold = 0.05` sits
//! inside the inlier noise on purpose — is the only one that takes the fallback.
//! That is the same "ship the verdict rather than assume it" shape
//! `huber_test.rs` uses for `outliers_`.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log per the CubeCL-HIP F64 gap, D-07) — even though the
//! RANSAC engine is host-resident on every backend, the fixture's f64 arm is
//! still an f64 fixture and the suite reports the skip the same way its siblings
//! do. Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use std::cell::Cell;
use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::linear::ransac::{
    MinSamples, RansacCallbacks, RansacDriver, RansacLoss, RansacModel, RansacRegressor,
    RansacTrialBridge, RansacVerdict, TrialStatus,
};
use mlrs_algos::typestate::Fitted;
use mlrs_algos::model_selection::rng::NumpyRandomState;
use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::ransac_host::RansacHostEngine;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry — `gen_oracle.py`'s `RANSAC_N_SAMPLES` × `RANSAC_N_FEATURES`.
const N_SAMPLES: usize = 300;
const N_FEATURES: usize = 5;
/// Held-out rows — `gen_oracle.py`'s `RANSAC_N_TEST`.
const N_TEST: usize = 11;
/// The fixture's `random_state`, and therefore the MT19937 seed the Rust side
/// starts its draw sequence from.
const SEED: u32 = 42;

/// Coefficient band on the f64 arm. Both sides run the SAME algorithm on the
/// SAME sub-samples and both solve the consensus least-squares exactly, so the
/// only source of disagreement is the order the two sum a `d`-length dot
/// product in; the measured gap over the whole surface is `~1e-13`.
const BAND_F64: f64 = 1e-9;
/// f32 band. The design's bytes carry ~7 digits, and — unlike mlrs, which runs
/// the sub-sample solve in `f64` whatever the design's width — scipy's `lstsq`
/// runs it at the design's own precision. So the two solve the same system to
/// different precisions and the agreement is bounded by `f32`'s, not by either
/// solver's. The reference is fitted on the design AFTER its round-trip through
/// f32 (`gen_ransac`), so this covers the SOLVE's f32 sensitivity and not a
/// dtype mismatch in the inputs.
const BAND_F32: f64 = 2e-5;

/// Relative distance from `residual_threshold` the nearest row must keep for a
/// case's `inlier_mask_` to be gated for EXACT equality (module docs).
const MASK_EXACT_MARGIN: f64 = 1e-4;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn load<F: Pod>() -> OracleCase {
    let tag = match std::mem::size_of::<F>() {
        4 => "f32",
        8 => "f64",
        _ => unreachable!("ransac fixtures are f32/f64 only"),
    };
    let path = fixture(&format!("ransac_{tag}_seed{SEED}.npz"));
    load_npz(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
}

/// The stored design/target, in the fixture's own width.
fn as_f<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    match std::mem::size_of::<F>() {
        4 => bytemuck::cast_slice::<f32, F>(
            case.f32(name)
                .unwrap_or_else(|| panic!("fixture is missing '{name}'")),
        )
        .to_vec(),
        8 => bytemuck::cast_slice::<f64, F>(
            case.f64(name)
                .unwrap_or_else(|| panic!("fixture is missing '{name}'")),
        )
        .to_vec(),
        _ => unreachable!("ransac fixtures are f32/f64 only"),
    }
}

fn to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("ransac fixtures are f32/f64 only"),
    }
}

fn band<F: Pod>() -> f64 {
    match std::mem::size_of::<F>() {
        4 => BAND_F32,
        8 => BAND_F64,
        _ => unreachable!("ransac fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ band + band·|exp|`.
fn assert_band(got: &[f64], expected: &[f64], band: f64, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= band + band * e.abs(),
            "{what}: band failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (band={band:e})"
        );
    }
}

/// One fixture case: the ctor configuration behind a `*_<name>` key group.
#[derive(Clone, Copy)]
struct Case {
    /// Fixture key suffix (`coef_<name>`, `counters_<name>`, …).
    name: &'static str,
    min_samples: MinSamples,
    residual_threshold: Option<f64>,
    max_trials: usize,
    max_skips: f64,
    stop_n_inliers: f64,
    stop_score: f64,
    stop_probability: f64,
    loss: RansacLoss,
    base_fit_intercept: bool,
    sample_weight: bool,
    /// Which validity predicate the case installs, if any. The predicates
    /// themselves are the fixture's, transcribed in [`data_valid`] /
    /// [`model_valid`].
    predicate: Predicate,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Predicate {
    None,
    Data,
    Model,
}

impl Case {
    /// sklearn's defaults, which is what every case starts from.
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            min_samples: MinSamples::Auto,
            residual_threshold: None,
            max_trials: 100,
            max_skips: f64::INFINITY,
            stop_n_inliers: f64::INFINITY,
            stop_score: f64::INFINITY,
            stop_probability: 0.99,
            loss: RansacLoss::AbsoluteError,
            base_fit_intercept: true,
            sample_weight: false,
            predicate: Predicate::None,
        }
    }
    const fn min_samples(mut self, v: MinSamples) -> Self {
        self.min_samples = v;
        self
    }
    const fn residual_threshold(mut self, v: f64) -> Self {
        self.residual_threshold = Some(v);
        self
    }
    const fn max_trials(mut self, v: usize) -> Self {
        self.max_trials = v;
        self
    }
    const fn max_skips(mut self, v: f64) -> Self {
        self.max_skips = v;
        self
    }
    const fn stop_n_inliers(mut self, v: f64) -> Self {
        self.stop_n_inliers = v;
        self
    }
    const fn stop_score(mut self, v: f64) -> Self {
        self.stop_score = v;
        self
    }
    const fn stop_probability(mut self, v: f64) -> Self {
        self.stop_probability = v;
        self
    }
    const fn loss(mut self, v: RansacLoss) -> Self {
        self.loss = v;
        self
    }
    const fn no_intercept(mut self) -> Self {
        self.base_fit_intercept = false;
        self
    }
    const fn weighted(mut self) -> Self {
        self.sample_weight = true;
        self
    }
    const fn predicate(mut self, v: Predicate) -> Self {
        self.predicate = v;
        self
    }
}

/// A fresh pool for one fit. The host arm never touches it — it exists for the
/// device scan, which every test here leaves unselected (`device` defaults to
/// `Auto` and `RANSAC_DEVICE_MIN_WORK` keeps `Auto` on the host).
fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

/// The fixture's `is_data_valid`: `max|X_subset| < 2.0`.
///
/// The predicate receives the sub-sample's ROW INDICES rather than a gathered
/// copy (RANSAC-02), so it reads the design it was handed — which is what the
/// shim's bridge does too, as one numpy fancy index.
fn data_valid<F: Pod>(x: &[F], d: usize, idxs: &[i64]) -> RansacVerdict {
    let worst = idxs
        .iter()
        .flat_map(|&g| &x[g as usize * d..(g as usize + 1) * d])
        .map(|&v| to_f64(v).abs())
        .fold(0.0f64, f64::max);
    if worst < 2.0 {
        RansacVerdict::Valid
    } else {
        RansacVerdict::Invalid
    }
}

/// The fixture's `is_model_valid`: `max|coef_| < 3.0`.
fn model_valid(model: RansacModel<'_>, _idxs: &[i64]) -> RansacVerdict {
    let worst = model.coef.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if worst < 3.0 {
        RansacVerdict::Valid
    } else {
        RansacVerdict::Invalid
    }
}

/// Every case in the fixture, in the order `gen_ransac` writes them.
const CASES: &[Case] = &[
    Case::new("default"),
    // --- `loss`: THE string-valued parameter, both options ------------------
    Case::new("loss_abs").loss(RansacLoss::AbsoluteError),
    Case::new("loss_sq").loss(RansacLoss::SquaredError),
    // --- `min_samples`: all three forms -------------------------------------
    Case::new("ms_int").min_samples(MinSamples::Absolute(20)),
    Case::new("ms_frac").min_samples(MinSamples::Fraction(0.3)),
    Case::new("ms_one").min_samples(MinSamples::Absolute(1)),
    // --- `residual_threshold` ------------------------------------------------
    Case::new("rt_tight").residual_threshold(0.4),
    Case::new("rt_loose").residual_threshold(8.0),
    // --- `max_trials` ---------------------------------------------------------
    Case::new("trials_3").max_trials(3),
    Case::new("trials_500").max_trials(500),
    // --- `stop_probability` ---------------------------------------------------
    Case::new("stopprob_low").stop_probability(0.1),
    Case::new("stopprob_high").stop_probability(0.999999),
    Case::new("stopprob_one").stop_probability(1.0),
    // --- the two early-exit thresholds ---------------------------------------
    Case::new("stop_inliers").stop_n_inliers(200.0),
    Case::new("stop_score").stop_score(0.5),
    // --- `max_skips`: the ConvergenceWarning branch ---------------------------
    Case::new("max_skips")
        .residual_threshold(0.05)
        .max_skips(3.0),
    // --- `estimator`: the base LinearRegression's `fit_intercept` -------------
    Case::new("noint")
        .no_intercept()
        .min_samples(MinSamples::Absolute(N_FEATURES + 1)),
    // --- the two validity callbacks -------------------------------------------
    Case::new("data_valid").predicate(Predicate::Data),
    Case::new("model_valid").predicate(Predicate::Model),
    // --- `sample_weight`, crossed with what it interacts with -----------------
    Case::new("sw").weighted(),
    Case::new("sw_ms20")
        .weighted()
        .min_samples(MinSamples::Absolute(20)),
    Case::new("sw_sq").weighted().loss(RansacLoss::SquaredError),
    Case::new("sw_noint")
        .weighted()
        .no_intercept()
        .min_samples(MinSamples::Absolute(N_FEATURES + 1)),
];

/// Fit one case and assert every fitted attribute the fixture ships.
fn check_case<F>(oracle: &OracleCase, case: &Case)
where
    F: Float + CubeElement + Pod,
{
    let x = as_f::<F>(oracle, "X");
    let y = as_f::<F>(oracle, "y");
    let x_test = as_f::<F>(oracle, "X_test");
    let sw: Vec<f64> = oracle.expect_f64("sample_weight").to_vec();

    let est = RansacRegressor::<F>::builder()
        .min_samples(case.min_samples)
        .residual_threshold(case.residual_threshold)
        .max_trials(case.max_trials)
        .max_skips(case.max_skips)
        .stop_n_inliers(case.stop_n_inliers)
        .stop_score(case.stop_score)
        .stop_probability(case.stop_probability)
        .loss(case.loss)
        .base_fit_intercept(case.base_fit_intercept)
        .build::<F>()
        .expect("the fixture's configurations are all inside sklearn's bounds");

    let dv = |idxs: &[i64]| data_valid::<F>(&x, N_FEATURES, idxs);
    let driver = match case.predicate {
        Predicate::None => RansacDriver::default(),
        Predicate::Data => RansacDriver::with_callbacks(RansacCallbacks {
            is_data_valid: Some(&dv),
            is_model_valid: None,
        }),
        Predicate::Model => RansacDriver::with_callbacks(RansacCallbacks {
            is_data_valid: None,
            is_model_valid: Some(&model_valid),
        }),
    };

    // The SAME seed sklearn's `check_random_state(42)` produces, so the two draw
    // the same sub-samples in the same order (module docs).
    let mut rng = NumpyRandomState::from_seed(SEED);
    let fitted = est
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            case.sample_weight.then_some(sw.as_slice()),
            &mut rng,
            &driver,
        )
        .unwrap_or_else(|e| panic!("case '{}': fit failed: {e}", case.name));

    let nm = case.name;
    let b = band::<F>();

    // --- the lowerings, gated as numbers -----------------------------------
    let exp_ms = oracle.expect_f64(&format!("min_samples_{nm}"))[0] as usize;
    assert_eq!(
        fitted.min_samples_used(),
        exp_ms,
        "case '{nm}': resolved min_samples"
    );
    let exp_rt = oracle.expect_f64(&format!("residual_threshold_{nm}"))[0];
    assert_band(
        &[fitted.residual_threshold_used()],
        &[exp_rt],
        1e-12,
        &format!("case '{nm}': resolved residual_threshold"),
    );

    // --- the TRAJECTORY: trial count and all three skip counters ------------
    let counters = oracle.expect_f64(&format!("counters_{nm}"));
    let got = [
        fitted.n_trials(),
        fitted.n_skips_no_inliers(),
        fitted.n_skips_invalid_data(),
        fitted.n_skips_invalid_model(),
    ];
    let want = [
        counters[0] as usize,
        counters[1] as usize,
        counters[2] as usize,
        counters[3] as usize,
    ];
    assert_eq!(
        got, want,
        "case '{nm}': (n_trials_, n_skips_no_inliers_, n_skips_invalid_data_, \
         n_skips_invalid_model_) diverged from sklearn's — the two did NOT walk \
         the same draw sequence"
    );

    // --- the consensus set ---------------------------------------------------
    let exp_mask = oracle.expect_f64(&format!("inlier_mask_{nm}"));
    let margin = oracle.expect_f64(&format!("margin_{nm}"))[0];
    let got_mask = fitted.inlier_mask();
    assert_eq!(got_mask.len(), exp_mask.len(), "case '{nm}': mask length");
    if margin > MASK_EXACT_MARGIN {
        let diff = got_mask
            .iter()
            .zip(exp_mask.iter())
            .position(|(&g, &e)| g != (e != 0.0));
        assert!(
            diff.is_none(),
            "case '{nm}': inlier_mask_ differs first at row {} (margin {margin:.3e} \
             says the comparison is well-posed)",
            diff.unwrap()
        );
    } else {
        // The fixture measured a row within reach of the residual comparison's
        // own rounding, so only the SIZE of the consensus is well-posed here
        // (module docs).
        let got_n = got_mask.iter().filter(|&&m| m).count();
        let want_n = exp_mask.iter().filter(|&&m| m != 0.0).count();
        assert_eq!(
            got_n, want_n,
            "case '{nm}': consensus SIZE (the exact mask is not gated at margin \
             {margin:.3e})"
        );
    }

    // --- the refitted base model --------------------------------------------
    assert_band(
        fitted.coef(),
        oracle.expect_f64(&format!("coef_{nm}")),
        b,
        &format!("case '{nm}': coef_"),
    );
    assert_band(
        fitted.intercept(),
        oracle.expect_f64(&format!("intercept_{nm}")),
        b,
        &format!("case '{nm}': intercept_"),
    );
    if !case.base_fit_intercept {
        assert_eq!(
            fitted.intercept(),
            &[0.0],
            "case '{nm}': fit_intercept=False must pin intercept_ to exactly 0"
        );
    }

    // --- predict on the held-out rows ---------------------------------------
    let pred = fitted
        .predict_from_host(&x_test, (N_TEST, N_FEATURES))
        .unwrap_or_else(|e| panic!("case '{nm}': predict failed: {e}"));
    assert!(
        pred.operand_finite,
        "case '{nm}': the fixture's held-out rows are finite, so the predict \
         pass must say so"
    );
    let pred64: Vec<f64> = pred.values.iter().map(|&v| to_f64(v)).collect();
    let exp_pred: Vec<f64> = match std::mem::size_of::<F>() {
        4 => oracle
            .f32(&format!("pred_{nm}"))
            .expect("fixture pred")
            .iter()
            .map(|&v| v as f64)
            .collect(),
        _ => oracle.expect_f64(&format!("pred_{nm}")).to_vec(),
    };
    assert_band(&pred64, &exp_pred, b, &format!("case '{nm}': predict"));
}

#[test]
fn full_parameter_surface_matches_sklearn_f32() {
    let oracle = load::<f32>();
    for case in CASES {
        check_case::<f32>(&oracle, case);
    }
}

#[test]
fn full_parameter_surface_matches_sklearn_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let oracle = load::<f64>();
    for case in CASES {
        check_case::<f64>(&oracle, case);
    }
}

/// `loss` is the ONE string-valued parameter, and this is its parse.
///
/// The Rust surface takes the typed [`RansacLoss`] and, at the PyO3 boundary
/// where the value arrives as text, the builder's `loss_str`. Both spellings
/// round-trip through [`RansacLoss::as_str`], and an unknown one becomes
/// [`BuildError::UnknownLoss`] — the single mapper the boundary already turns
/// into sklearn's `ValueError` (D-09), rather than a silent fallback to the
/// default.
#[test]
fn loss_strings_round_trip_and_reject_the_unknown() {
    for (text, variant) in [
        ("absolute_error", RansacLoss::AbsoluteError),
        ("squared_error", RansacLoss::SquaredError),
    ] {
        let built = RansacRegressor::<f32>::builder()
            .loss_str(text)
            .expect("a documented sklearn option must parse")
            .build::<f32>()
            .expect("defaults are valid");
        assert_eq!(built.loss(), variant, "loss_str({text:?}) parsed wrong");
        assert_eq!(
            variant.as_str(),
            text,
            "as_str does not round-trip {text:?}"
        );
    }
    let err = RansacRegressor::<f32>::builder()
        .loss_str("huber")
        .expect_err("an option sklearn does not offer must be rejected");
    assert!(
        matches!(err, BuildError::UnknownLoss { ref value } if value == "huber"),
        "expected UnknownLoss(\"huber\"), got {err:?}"
    );
}

/// The two `loss` options must actually SELECT different arithmetic.
///
/// A `loss=` that were quietly inert would still pass every fixture case whose
/// consensus happens to coincide, so this pins the difference directly: at a
/// threshold between `|r|` and `r²` for part of the design, the two options
/// cannot classify the same rows.
#[test]
fn the_two_loss_options_classify_differently() {
    let oracle = load::<f64>();
    if capability::skip_f64_with_log() {
        return;
    }
    let x = as_f::<f64>(&oracle, "X");
    let y = as_f::<f64>(&oracle, "y");
    let mut sizes = Vec::new();
    for loss in [RansacLoss::AbsoluteError, RansacLoss::SquaredError] {
        let mut rng = NumpyRandomState::from_seed(SEED);
        let fitted = RansacRegressor::<f64>::builder()
            .loss(loss)
            // `|r| <= t` and `r^2 <= t` are the same test only at `t = 1`;
            // below it the squared form is the more permissive of the two by a
            // factor of `sqrt(t)/t`. At `0.05` against this fixture's 0.1-sigma
            // inlier noise that is the difference between keeping the rows
            // inside half a sigma and keeping the rows inside two, so the two
            // consensus sets cannot coincide.
            .residual_threshold(Some(0.05))
            .build::<f64>()
            .expect("valid")
            .fit_from_host_slice(
                &mut pool(),
                &x,
                &y,
                (N_SAMPLES, N_FEATURES),
                1,
                None,
                &mut rng,
                &RansacDriver::default(),
            )
            .expect("fit");
        sizes.push(fitted.inlier_mask().iter().filter(|&&m| m).count());
    }
    assert_ne!(
        sizes[0], sizes[1],
        "absolute_error and squared_error produced the same consensus size \
         ({sizes:?}) at residual_threshold=0.05 — the loss parameter is inert"
    );
}

/// A predicate that answers [`RansacVerdict::Abort`] stops the fit.
///
/// This is the escape hatch that keeps the Rust core free of any foreign error
/// type: a PyO3 wrapper whose Python callable raised stashes the real `PyErr`,
/// answers `Abort`, and re-raises after the unwind. Here the same mechanism is
/// exercised without Python — the third trial aborts, and the error names which
/// predicate did it.
#[test]
fn a_callback_can_abort_the_fit() {
    let oracle = load::<f32>();
    let x = as_f::<f32>(&oracle, "X");
    let y = as_f::<f32>(&oracle, "y");

    let calls = Cell::new(0usize);
    let abort_on_third = |_idxs: &[i64]| -> RansacVerdict {
        calls.set(calls.get() + 1);
        if calls.get() >= 3 {
            RansacVerdict::Abort
        } else {
            RansacVerdict::Valid
        }
    };
    let err = RansacRegressor::<f32>::builder()
        .build::<f32>()
        .expect("valid")
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            &RansacDriver::with_callbacks(RansacCallbacks {
                is_data_valid: Some(&abort_on_third),
                is_model_valid: None,
            }),
        )
        .expect_err("an aborting predicate must not produce a fitted estimator");
    assert!(
        matches!(
            err,
            AlgoError::CallbackAborted {
                callback: "is_data_valid",
                ..
            }
        ),
        "expected CallbackAborted(is_data_valid), got {err:?}"
    );
    assert_eq!(calls.get(), 3, "the fit ran past the aborting trial");
}

/// A predicate that rejects EVERYTHING leaves no consensus set, which is
/// sklearn's `ValueError` and not a warning: with no best sub-sample there is
/// nothing to refit.
///
/// The two branches of that error are distinguished by whether the skip budget
/// was blown, and `skipped_out` carries the choice so the PyO3 layer can
/// reproduce sklearn's exact wording — both are pinned here.
#[test]
fn rejecting_every_subsample_is_an_error_not_a_warning() {
    let oracle = load::<f32>();
    let x = as_f::<f32>(&oracle, "X");
    let y = as_f::<f32>(&oracle, "y");
    let never = |_idxs: &[i64]| RansacVerdict::Invalid;

    for (max_skips, want_skipped_out, want_trials) in
        [(f64::INFINITY, false, 100usize), (3.0, true, 5usize)]
    {
        let err = RansacRegressor::<f32>::builder()
            .max_skips(max_skips)
            .build::<f32>()
            .expect("valid")
            .fit_from_host_slice(
                &mut pool(),
                &x,
                &y,
                (N_SAMPLES, N_FEATURES),
                1,
                None,
                &mut NumpyRandomState::from_seed(SEED),
                &RansacDriver::with_callbacks(RansacCallbacks {
                    is_data_valid: Some(&never),
                    is_model_valid: None,
                }),
            )
            .expect_err("no consensus set must be an error");
        match err {
            AlgoError::NoValidConsensusSet {
                n_trials,
                skipped_out,
                ..
            } => {
                assert_eq!(
                    skipped_out, want_skipped_out,
                    "max_skips={max_skips}: wrong sklearn message branch"
                );
                assert_eq!(
                    n_trials, want_trials,
                    "max_skips={max_skips}: wrong trial count before giving up"
                );
            }
            other => panic!("expected NoValidConsensusSet, got {other:?}"),
        }
    }
}

/// The `max_skips` case found a consensus DESPITE blowing its budget — sklearn
/// warns there rather than raising, so mlrs surfaces the fact as a flag on the
/// fitted estimator and the shim turns it into the `ConvergenceWarning`.
#[test]
fn a_blown_skip_budget_with_a_consensus_is_reported_not_raised() {
    let oracle = load::<f32>();
    let x = as_f::<f32>(&oracle, "X");
    let y = as_f::<f32>(&oracle, "y");
    let fitted = RansacRegressor::<f32>::builder()
        .residual_threshold(Some(0.05))
        .max_skips(3.0)
        .build::<f32>()
        .expect("valid")
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            &RansacDriver::default(),
        )
        .expect("a consensus WAS found, so this is a warning case not an error");
    assert!(
        fitted.exceeded_max_skips(),
        "the fixture's `max_skips` case blows its budget; the flag must say so"
    );
    // And the default case must NOT set it, or the flag would be vacuous.
    let clean = RansacRegressor::<f32>::builder()
        .build::<f32>()
        .expect("valid")
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            &RansacDriver::default(),
        )
        .expect("fit");
    assert!(
        !clean.exceeded_max_skips(),
        "the default fit blew no budget"
    );
}

/// `min_samples > n_samples` is a data-DEPENDENT rejection at `fit`, not a
/// builder one — the resolved value is `n_features + 1` or
/// `ceil(frac·n_samples)`, neither knowable at `build()` (the D-08 split).
#[test]
fn min_samples_larger_than_the_design_is_rejected_at_fit() {
    let oracle = load::<f32>();
    let x = as_f::<f32>(&oracle, "X");
    let y = as_f::<f32>(&oracle, "y");
    let err = RansacRegressor::<f32>::builder()
        .min_samples(MinSamples::Absolute(N_SAMPLES + 1))
        .build::<f32>()
        .expect("the bound is not knowable at build time, so this must build")
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            &RansacDriver::default(),
        )
        .expect_err("a sub-sample larger than the design must be rejected");
    assert!(
        matches!(
            err,
            AlgoError::MinSamplesExceedsNSamples {
                min_samples,
                n_samples,
                ..
            } if min_samples == N_SAMPLES + 1 && n_samples == N_SAMPLES
        ),
        "expected MinSamplesExceedsNSamples, got {err:?}"
    );
}

/// The data-INDEPENDENT bounds, rejected at `build()` (the other half of D-08).
///
/// These are sklearn's `_parameter_constraints`, with the one deliberate
/// addition the builder documents: sklearn ADMITS `min_samples=0.0` and then
/// dies on an unbound local inside its own `fit`, so mlrs rejects it here with
/// a message that names the bound.
#[test]
fn builder_rejects_out_of_range_hyperparameters() {
    let cases: Vec<(&str, Box<dyn Fn() -> Result<_, BuildError>>)> = vec![
        (
            "max_trials",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .max_trials(0)
                    .build::<f32>()
            }),
        ),
        (
            "max_skips",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .max_skips(-1.0)
                    .build::<f32>()
            }),
        ),
        (
            "stop_n_inliers",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .stop_n_inliers(-1.0)
                    .build::<f32>()
            }),
        ),
        (
            "stop_probability",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .stop_probability(1.5)
                    .build::<f32>()
            }),
        ),
        (
            "residual_threshold",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .residual_threshold(Some(-1.0))
                    .build::<f32>()
            }),
        ),
        (
            "min_samples",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .min_samples(MinSamples::Absolute(0))
                    .build::<f32>()
            }),
        ),
        (
            "min_samples",
            Box::new(|| {
                RansacRegressor::<f32>::builder()
                    .min_samples(MinSamples::Fraction(0.0))
                    .build::<f32>()
            }),
        ),
    ];
    for (param, build) in cases {
        let err = build().expect_err("{param}: out-of-range value must be rejected");
        assert!(
            matches!(err, BuildError::InvalidHyperprior { param: p, .. } if p == param),
            "expected InvalidHyperprior({param}), got {err:?}"
        );
    }
    // And the defaults themselves must build — a validator that rejected them
    // would make every assert above pass for the wrong reason.
    RansacRegressor::<f32>::builder()
        .build::<f32>()
        .expect("sklearn's defaults must be inside sklearn's bounds");
}

/// The consensus scan is split across a worker pool, and its floating-point
/// result must not depend on how wide that pool is.
///
/// This is not a style preference: the tie-break between two sub-samples with
/// the SAME inlier count compares their R², so a reduction that reassociated
/// with the thread count could pick a different final model on a different
/// machine. `ransac_host` blocks the row axis at a fixed size for exactly this
/// reason, and this test is what holds it to that.
///
/// The width is pinned through the `MLRS_RANSAC_UNITS` abflag rather than the
/// environment, so the override is scoped to this thread and does not race the
/// sibling tests libtest runs in parallel ([[mlrs-abflag-test-knobs]]).
#[test]
fn the_fit_is_identical_at_every_worker_width() {
    let oracle = load::<f32>();
    let x = as_f::<f32>(&oracle, "X");
    let y = as_f::<f32>(&oracle, "y");

    let run = || {
        let mut rng = NumpyRandomState::from_seed(SEED);
        let fitted = RansacRegressor::<f32>::builder()
            .build::<f32>()
            .expect("valid")
            .fit_from_host_slice(
                &mut pool(),
                &x,
                &y,
                (N_SAMPLES, N_FEATURES),
                1,
                None,
                &mut rng,
                &RansacDriver::default(),
            )
            .expect("fit");
        (
            fitted.coef().to_vec(),
            fitted.n_trials(),
            fitted.inlier_mask().to_vec(),
        )
    };

    let mut reference: Option<(Vec<f64>, usize, Vec<bool>)> = None;
    for units in ["1", "2", "3", "7"] {
        let _guard = mlrs_backend::abflag::force("MLRS_RANSAC_UNITS", units);
        let got = run();
        match &reference {
            None => reference = Some(got),
            Some(want) => assert_eq!(
                &got, want,
                "the fit changed at MLRS_RANSAC_UNITS={units} — the scan's \
                 reduction is not worker-count independent"
            ),
        }
    }
}

// =========================================================================== //
// The arbitrary-base arm (RANSAC-02)
// =========================================================================== //

/// A [`RansacTrialBridge`] over the SAME least squares the native arm fits, so
/// the two arms are comparable statement for statement.
///
/// It exists to test the CONTRACT rather than any particular estimator: a
/// foreign base's only job is to answer "here are my predictions for the whole
/// design, or here is why I am skipping", and everything the loop does with that
/// answer is what these tests are about. The PyO3 implementation is the same
/// shape with `estimator.fit`/`predict` in place of `subset_lstsq`.
struct OlsBridge<'a> {
    engine: RansacHostEngine<'a, f64>,
    x: &'a [f64],
    n: usize,
    d: usize,
    /// How many times the loop called in — the per-trial crossing count, which
    /// is the claim the foreign arm is built around.
    calls: Cell<usize>,
    /// A trial index whose data the bridge rejects, and one whose model it does.
    reject_data_at: Option<usize>,
    reject_model_at: Option<usize>,
    /// The consensus refit, so the test can check the loop asked for one.
    refits: Cell<usize>,
}

impl OlsBridge<'_> {
    fn seen(&self) -> usize {
        self.calls.get()
    }
}

impl RansacTrialBridge for OlsBridge<'_> {
    fn run_trial(
        &self,
        idxs: &[i64],
        _rng: &mut NumpyRandomState,
        scan: &mut dyn FnMut(&[f64], Option<&[f64]>),
    ) -> Result<TrialStatus, ()> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.reject_data_at == Some(call) {
            return Ok(TrialStatus::InvalidData);
        }
        let (coef, icept) = self.engine.subset_lstsq(idxs, true, None);
        if self.reject_model_at == Some(call) {
            return Ok(TrialStatus::InvalidModel);
        }
        let y_pred: Vec<f64> = (0..self.n)
            .map(|i| {
                let row = &self.x[i * self.d..(i + 1) * self.d];
                row.iter().zip(&coef).map(|(a, b)| a * b).sum::<f64>() + icept[0]
            })
            .collect();
        // The scan reads this buffer in place and does not outlive the call —
        // the callback contract this trait is shaped around.
        scan(&y_pred, None);
        Ok(TrialStatus::Fitted)
    }

    fn refit(&self, _idxs: &[i64], _rng: &mut NumpyRandomState) -> Result<(), ()> {
        self.refits.set(self.refits.get() + 1);
        Ok(())
    }
}

fn ols_bridge<'a>(x: &'a [f64], y: &'a [f64]) -> OlsBridge<'a> {
    OlsBridge {
        engine: RansacHostEngine::new(x, y, N_SAMPLES, N_FEATURES, 1, N_SAMPLES * N_FEATURES)
            .expect("geometry"),
        x,
        n: N_SAMPLES,
        d: N_FEATURES,
        calls: Cell::new(0),
        reject_data_at: None,
        reject_model_at: None,
        refits: Cell::new(0),
    }
}

fn fit_default(x: &[f64], y: &[f64], driver: &RansacDriver<'_>) -> RansacRegressor<f64, Fitted> {
    RansacRegressor::<f64>::builder()
        .build::<f64>()
        .expect("sklearn's defaults are in range")
        .fit_from_host_slice(
            &mut pool(),
            x,
            y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            driver,
        )
        .expect("the fixture design has a consensus set")
}

/// A base the loop cannot fit itself produces the SAME fit as the one it can,
/// for one call per trial and no more.
#[test]
fn a_foreign_base_matches_the_native_arm() {
    if capability::skip_f64_with_log() {
        return;
    }
    let oracle = load::<f64>();
    let (x, y) = (as_f::<f64>(&oracle, "X"), as_f::<f64>(&oracle, "y"));

    let native = fit_default(&x, &y, &RansacDriver::default());
    let bridge = ols_bridge(&x, &y);
    let foreign = fit_default(&x, &y, &RansacDriver::foreign(&bridge));

    assert_eq!(native.n_trials(), foreign.n_trials(), "n_trials_");
    assert_eq!(
        native.inlier_mask(),
        foreign.inlier_mask(),
        "the consensus set"
    );
    assert_eq!(
        native.n_skips_no_inliers(),
        foreign.n_skips_no_inliers(),
        "n_skips_no_inliers_"
    );
    // ONE call per trial — the floor the module docs claim, measured rather
    // than asserted in prose.
    assert_eq!(
        bridge.seen(),
        foreign.n_trials(),
        "the loop must call the bridge exactly once per trial"
    );
    assert_eq!(bridge.refits.get(), 1, "exactly one consensus refit");
    // The foreign arm holds no linear model of its own: the fitted estimator is
    // the caller's object, and asking this one to predict is an error rather
    // than a silently empty matvec.
    assert!(!foreign.has_linear_model());
    assert!(foreign.predict_from_host(&x, (N_SAMPLES, N_FEATURES)).is_err());
    assert_eq!(foreign.device_arm(), "cpu", "the foreign arm has no device");
}

/// The bridge's two skip verdicts land in sklearn's two counters.
#[test]
fn a_foreign_base_can_skip_a_trial_either_way() {
    if capability::skip_f64_with_log() {
        return;
    }
    let oracle = load::<f64>();
    let (x, y) = (as_f::<f64>(&oracle, "X"), as_f::<f64>(&oracle, "y"));

    let mut bridge = ols_bridge(&x, &y);
    bridge.reject_data_at = Some(0);
    bridge.reject_model_at = Some(1);
    let fitted = fit_default(&x, &y, &RansacDriver::foreign(&bridge));

    assert_eq!(fitted.n_skips_invalid_data(), 1, "n_skips_invalid_data_");
    assert_eq!(fitted.n_skips_invalid_model(), 1, "n_skips_invalid_model_");
}

/// A bridge that gives up unwinds the fit as a callback abort, and does not
/// leave a half-fitted estimator behind.
#[test]
fn a_foreign_base_can_abort_the_fit() {
    if capability::skip_f64_with_log() {
        return;
    }
    let oracle = load::<f64>();
    let (x, y) = (as_f::<f64>(&oracle, "X"), as_f::<f64>(&oracle, "y"));

    struct Boom;
    impl RansacTrialBridge for Boom {
        fn run_trial(
            &self,
            _idxs: &[i64],
            _rng: &mut NumpyRandomState,
            _scan: &mut dyn FnMut(&[f64], Option<&[f64]>),
        ) -> Result<TrialStatus, ()> {
            Err(())
        }
        fn refit(&self, _idxs: &[i64], _rng: &mut NumpyRandomState) -> Result<(), ()> {
            Ok(())
        }
    }

    let boom = Boom;
    let err = RansacRegressor::<f64>::builder()
        .build::<f64>()
        .expect("defaults")
        .fit_from_host_slice(
            &mut pool(),
            &x,
            &y,
            (N_SAMPLES, N_FEATURES),
            1,
            None,
            &mut NumpyRandomState::from_seed(SEED),
            &RansacDriver::foreign(&boom),
        )
        .expect_err("a bridge that gives up must not produce a fitted estimator");
    match err {
        AlgoError::CallbackAborted { callback, .. } => {
            assert_eq!(callback, "estimator.fit")
        }
        other => panic!("expected CallbackAborted, got {other:?}"),
    }
}
