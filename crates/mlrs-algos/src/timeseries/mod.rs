//! `timeseries` — time-series estimators (TSA-01, Phase 22).
//!
//! Module index. `Arima` is the ARIMA(p,d,q) core (Kalman-filter MLE,
//! forecast); `AutoArima` wraps it with a bounded `(p,d,q)` grid search over
//! AICc. See `arima`'s module docs for the full scope statement.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod arima;

pub use arima::{Arima, AutoArima};
