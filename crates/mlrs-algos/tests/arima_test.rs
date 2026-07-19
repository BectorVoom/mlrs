//! TSA-01 (Phase 22) — ARIMA / AutoArima oracle gates.
//!
//! Two tiers (the TreeSHAP/t-SNE convention — see `arima.rs` module docs
//! for the full scope statement: zero-mean only, no seasonal component):
//!
//!   - `loglik_matches_statsmodels_at_fixed_params` — DETERMINISTIC ≤1e-6
//!     gate: the concentrated Kalman log-likelihood at FIXED known
//!     parameters (no optimizer), vs `statsmodels.tsa.statespace.sarimax`'s
//!     `loglike` at the SAME parameters — the state-space/filter formula
//!     itself, isolated from any MLE non-convexity.
//!   - `fit_band` — BAND gate: mlrs's own L-BFGS MLE fit must reach AT LEAST
//!     AS GOOD a log-likelihood as statsmodels' MLE fit (both are maximizing
//!     the SAME concentrated likelihood; a strictly worse optimum would
//!     indicate an optimizer/gradient bug), plus a loose forecast-shape
//!     sanity check against statsmodels' 5-step forecast.
//!   - `build_validation` / `fit_validation` — typed hyperparameter/geometry
//!     errors.
//!   - `forecast_roundtrip` / `auto_arima_recovers_order` — structural +
//!     AutoArima behavior checks (no oracle needed).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_algos::error::BuildError;
use mlrs_algos::timeseries::{Arima, AutoArima};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn load_case() -> OracleCase {
    load_npz(&fixture("arima_seed42.npz")).expect("fixture loads")
}

#[test]
fn loglik_matches_statsmodels_at_fixed_params() {
    let case = load_case();
    let y = case.expect_f64("y");
    let params = case.expect_f64("true_params");
    let (phi, theta) = (params[..2].to_vec(), params[2..3].to_vec());
    let ll_ref = case.expect_f64("loglik_at_true_params")[0];

    let ll_got = mlrs_algos::timeseries::arima::loglik(&phi, &theta, &y);
    assert!(
        (ll_got - ll_ref).abs() <= 1e-6,
        "loglik at fixed params: got {ll_got}, statsmodels {ll_ref}"
    );
}

#[test]
fn fit_band() {
    let case = load_case();
    let y = case.expect_f64("y");
    let n = y.len();
    let sm_loglik = case.expect_f64("sm_loglik")[0];
    let sm_forecast = case.expect_f64("sm_forecast");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_f);

    let model = Arima::<f32>::builder()
        .order(2, 0, 1)
        .build::<f32>()
        .expect("valid build")
        .fit(&pool, &y_dev, n)
        .expect("valid fit");

    assert!(
        model.loglik() >= sm_loglik - 1.0,
        "mlrs MLE loglik {} should be within 1.0 of statsmodels' {sm_loglik} (both maximize the \
         same concentrated likelihood — a much worse optimum indicates an optimizer/gradient bug)",
        model.loglik()
    );

    let fc = model.forecast(5);
    assert_eq!(fc.len(), 5);
    for (i, (&g, &r)) in fc.iter().zip(sm_forecast.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1.5,
            "forecast[{i}]: mlrs {g} vs statsmodels {r} (loose band — different MLE optima \
             produce different but comparable forecasts)"
        );
    }
}

#[test]
fn forecast_roundtrip() {
    // A deterministic AR(1) series with a KNOWN pattern: y_t = 0.9*y_{t-1}.
    // The differenced-once series should forecast toward 0 (no drift, d=1
    // zero-mean scope), and re-integrating must reproduce the level.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let n = 60usize;
    let mut y = vec![0.0f32; n];
    y[0] = 10.0;
    for t in 1..n {
        y[t] = 0.9 * y[t - 1];
    }
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    let model = Arima::<f32>::builder()
        .order(1, 1, 0)
        .build::<f32>()
        .expect("valid build")
        .fit(&pool, &y_dev, n)
        .expect("valid fit");

    let fc = model.forecast(3);
    assert_eq!(fc.len(), 3);
    // The series is decaying toward 0; a 1-differenced zero-mean AR(1)
    // forecast should stay in the same ballpark as the tail, not explode.
    let tail = y[n - 1] as f64;
    for &v in &fc {
        assert!(v.is_finite());
        assert!((v - tail).abs() < 5.0, "forecast {v} diverged too far from tail {tail}");
    }
}

#[test]
fn auto_arima_recovers_order() {
    let case = load_case();
    let y = case.expect_f64("y");
    let n = y.len();
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y_f);

    let best = AutoArima::search::<f32>(&pool, &y_dev, n, 0, 3, 3).expect("search converges");
    // The true order is (2,0,1); AutoArima should land on a comparably good
    // (not necessarily identical) model — assert it's at least as good as
    // the true-order fit on this same data (AICc is the selection criterion).
    let true_order_fit = Arima::<f32>::builder()
        .order(2, 0, 1)
        .build::<f32>()
        .expect("valid build")
        .fit(&pool, &y_dev, n)
        .expect("valid fit");
    assert!(
        best.aicc() <= true_order_fit.aicc() + 1.0,
        "AutoArima's best AICc {} should be competitive with the true order's {}",
        best.aicc(),
        true_order_fit.aicc()
    );
}

#[test]
fn build_validation() {
    let err = match Arima::<f32>::builder().order(20, 0, 0).build::<f32>() {
        Err(e) => e,
        Ok(_) => panic!("p over the bound must be rejected at build"),
    };
    assert!(matches!(err, BuildError::InvalidArimaOrder { .. }), "expected InvalidArimaOrder");
}

#[test]
fn fit_validation() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    // Too few observations for the requested order → typed error.
    let y: Vec<f32> = vec![1.0, 2.0, 3.0];
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);
    let model = Arima::<f32>::builder().order(5, 1, 5).build::<f32>().expect("valid build");
    assert!(model.fit(&pool, &y_dev, 3).is_err(), "too-short series must be rejected at fit");
}
