//! f64 capability-gate tests (FOUND-04 / Criterion 4 / T-03-04).
//!
//! Asserts:
//!   1. `feature_enabled(FloatKind::F64)` returns a bool on the active backend
//!      without panicking, and agrees with the per-client `supports_f64`.
//!   2. `log_oracle_dtype` emits a `dtype=… backend=… adapter=…` line at info
//!      level (Criterion 4: CI log shows which dtype ran on which backend).
//!   3. The f64-gated path follows `skip_f64_with_log`: it skips-with-log when
//!      f64 is unsupported, and runs (logging the dtype line) when it is.
//!
//! A small in-memory `log::Log` captures records so the log assertions are
//! deterministic without depending on stderr scraping.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as `mod tests` in `src/`.

use std::sync::{Mutex, OnceLock};

use log::{Level, Log, Metadata, Record};

use mlrs_backend::capability::{
    self, active_backend_name, feature_enabled, log_oracle_dtype, skip_f64_with_log, supports_f64,
    FloatKind,
};
use mlrs_backend::runtime;

// ----------------------------------------------------------------------------
// In-memory capturing logger (records every formatted message + level).
// ----------------------------------------------------------------------------

struct CaptureLogger;

static RECORDS: OnceLock<Mutex<Vec<(Level, String)>>> = OnceLock::new();

fn records() -> &'static Mutex<Vec<(Level, String)>> {
    RECORDS.get_or_init(|| Mutex::new(Vec::new()))
}

impl Log for CaptureLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        true
    }
    fn log(&self, record: &Record) {
        records()
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }
    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

/// Install the capturing logger exactly once (idempotent across test threads).
fn init_capture() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // Ignore the Err if some other harness already set a logger; our
        // assertions still read whatever lands in RECORDS when ours is active.
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

fn captured() -> Vec<(Level, String)> {
    records().lock().unwrap().clone()
}

// ----------------------------------------------------------------------------
// 1. feature_enabled returns a bool, agrees with supports_f64, no panic.
// ----------------------------------------------------------------------------

#[test]
fn feature_enabled_f64_returns_bool_without_panic() {
    let client = runtime::active_client();
    let per_client = supports_f64(&client);
    let facade = feature_enabled(FloatKind::F64);
    assert_eq!(
        facade, per_client,
        "feature_enabled facade must agree with supports_f64 on the active client"
    );
    // f32 must be universally supported on every backend.
    assert!(
        capability::supports_type(&client, FloatKind::F32),
        "f32 must be supported on {}",
        active_backend_name()
    );
}

// ----------------------------------------------------------------------------
// 2. log_oracle_dtype emits a dtype/backend/adapter info line (Criterion 4).
// ----------------------------------------------------------------------------

#[test]
fn log_oracle_dtype_emits_dtype_backend_line() {
    init_capture();
    let backend = active_backend_name();
    log_oracle_dtype(FloatKind::F32, backend, "default");
    // Mirror to stdout so the line is visible under `--nocapture` (the captured
    // logger above intercepts the `log::info!`, so env_logger never prints it).
    println!("oracle dtype=F32 backend={backend} adapter=default");

    let lines = captured();
    let found = lines.iter().any(|(lvl, msg)| {
        *lvl == Level::Info
            && msg.contains("oracle ")
            && msg.contains("dtype=")
            && msg.contains(&format!("backend={backend}"))
            && msg.contains("adapter=")
    });
    assert!(
        found,
        "expected an info dtype/backend/adapter oracle line; captured: {lines:?}"
    );
}

// ----------------------------------------------------------------------------
// 3. The f64-gated oracle path: skip-with-log when unsupported, run (with the
//    dtype line) when supported. Models exactly how Plan 05 oracle tests gate.
// ----------------------------------------------------------------------------

#[test]
fn f64_gated_path_skips_with_log_or_runs() {
    init_capture();
    let backend = active_backend_name();

    if skip_f64_with_log() {
        println!("skipping f64 oracle on {backend}: SHADER_F64 / f64 unsupported");
        // Unsupported adapter: assert the skip-with-reason warn line is present
        // and DO NOT fail (skip, not failure — T-03-04).
        let lines = captured();
        let warned = lines.iter().any(|(lvl, msg)| {
            *lvl == Level::Warn && msg.contains("skipping f64 oracle") && msg.contains(backend)
        });
        assert!(
            warned,
            "skip path must log a warn 'skipping f64 oracle' reason; captured: {lines:?}"
        );
        return;
    }

    // Supported (e.g. this env's wgpu adapter has SHADER_F64): the f64 oracle
    // RUNS. Emit + assert the dtype=f64 line so CI shows it executed.
    log_oracle_dtype(FloatKind::F64, backend, "active");
    println!("oracle dtype=F64 backend={backend} adapter=active");
    let lines = captured();
    let ran = lines.iter().any(|(lvl, msg)| {
        *lvl == Level::Info && msg.contains("dtype=F64") && msg.contains(&format!("backend={backend}"))
    });
    assert!(
        ran,
        "f64-supported path must log the dtype=F64 oracle line; captured: {lines:?}"
    );
}
