//! CubeCL stream-count cap (STREAM-CAP-01).
//!
//! ## Why mlrs caps the stream count
//!
//! CubeCL allocates **one GPU stream per OS thread**: `StreamId::current()`
//! reads a thread-local drawn from a global counter, and `StreamPool` indexes
//! streams as `stream_id.value % max_streams`. `max_streams` defaults to
//! **128**, and every `Stream` owns an INDEPENDENT
//! `MemoryManagement<GpuStorage>` — a device page belongs to exactly one
//! stream's arena, and streams are never torn down.
//!
//! That default is wrong for mlrs in both directions:
//!
//! * **It cannot help.** Every device call in the Python extension runs while
//!   holding the process-global `Mutex<BufferPool>` (`crates/mlrs-py/src/lib.rs`),
//!   so mlrs never has two kernels in flight from two threads. Extra streams buy
//!   exactly zero overlap.
//! * **It costs a great deal.** Any thread fan-out over mlrs estimators (joblib's
//!   `threading` backend under `StackingRegressor(n_jobs=...)`, a threaded
//!   `GridSearchCV`, a user's own `ThreadPoolExecutor`) mints a fresh thread —
//!   and therefore a fresh stream and a fresh arena — per `Parallel` call.
//!   Buffers mlrs's own pool would have reused are re-allocated per stream, and
//!   every launch then pays cross-stream flush/wait alignment.
//!
//! On rocm gfx1151 — where the VRAM carve-out is 512 MB and already ~96% used at
//! idle — that reliably exhausted the device heap and killed the process:
//! `HipServer::initialize_memory`'s `command.reserve(size).unwrap()` failed on
//! the shared `DSD-0-0` server thread, and every subsequent command cascaded
//! into `"Memory page N doesn't exist"` / `CallError` panics. Measured on the
//! reproducer (`crates/mlrs-backend/tests/thread_stream_probe_test.rs` documents
//! the full investigation):
//!
//! | `max_streams` | outcome |
//! |---|---|
//! | 128 (cubecl default) | **aborts** deterministically, 3/3 runs |
//! | 16 / 8 / 4 / 2 / 1 | survives; results bit-identical to serial |
//!
//! Capping is also *faster* — the cross-stream alignment disappears (cv=5,
//! `n_jobs=4`: ~3.2 s → ~0.24 s).
//!
//! ## What this module does
//!
//! [`install`] runs once, before the first client is built, and writes the
//! CubeCL global config with `streaming.max_streams` capped to
//! [`MLRS_MAX_STREAMS`]. It is deliberately conservative about overriding a
//! deliberate choice:
//!
//! 1. `MLRS_MAX_STREAMS=<n>` in the environment wins outright (the escape hatch
//!    — set it to a large value to reproduce the old behaviour).
//! 2. Otherwise, a `cubecl.toml` that sets `streaming.max_streams` to anything
//!    other than cubecl's own default is the user's decision and is respected.
//! 3. Otherwise the cap applies.
//!
//! If some other code has already read or set the CubeCL config before the first
//! `active_client()` call, this cannot change it (cubecl's config is
//! write-once); it leaves the existing value alone and logs a warning if that
//! value is above the cap.
//!
//! Tests live in `crates/mlrs-backend/tests/stream_cap_test.rs` (AGENTS.md §2).

use std::sync::OnceLock;

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

/// The stream count mlrs installs when the caller has expressed no preference.
///
/// One: mlrs serializes every device call behind one mutex, so a second stream
/// can never carry concurrent work — it can only fragment the device arena.
pub const MLRS_MAX_STREAMS: u8 = 1;

/// cubecl's own `default_max_streams()`, duplicated because it is a private
/// function in `cubecl-runtime::config::streaming`.
///
/// Used ONLY to tell "the user configured this" from "this is the untouched
/// default". `stream_cap_test.rs` asserts it still matches
/// `CubeClRuntimeConfig::default()`, so a cubecl upgrade that changes the
/// default fails loudly instead of silently disabling rule 2 above.
pub const CUBECL_DEFAULT_MAX_STREAMS: u8 = 128;

/// Environment escape hatch. Set to an integer `>= 1` to choose the stream count
/// explicitly (e.g. `MLRS_MAX_STREAMS=128` restores cubecl's default).
pub const MAX_STREAMS_ENV: &str = "MLRS_MAX_STREAMS";

static INSTALLED: OnceLock<u8> = OnceLock::new();

/// Install the cap into CubeCL's global config, once per process.
///
/// Returns the `max_streams` actually in effect — which is NOT always
/// [`MLRS_MAX_STREAMS`]: see the module docs for the three precedence rules and
/// the already-initialized case.
///
/// Called from [`crate::runtime::active_client`] before the first client is
/// built. Idempotent and cheap after the first call.
pub fn install() -> u8 {
    *INSTALLED.get_or_init(install_once)
}

/// The `max_streams` in effect, or `None` if [`install`] has not run yet.
///
/// For tests and diagnostics: unlike [`install`], this does not initialize
/// anything, so a test can assert the cap was applied by the client
/// construction rather than by its own call.
pub fn installed() -> Option<u8> {
    INSTALLED.get().copied()
}

/// Parse the escape hatch. An unparseable or zero value is ignored (0 would
/// panic cubecl's `stream_id % max_streams`), with a warning.
fn env_override() -> Option<u8> {
    let raw = std::env::var(MAX_STREAMS_ENV).ok()?;
    match raw.trim().parse::<u8>() {
        Ok(n) if n >= 1 => Some(n),
        _ => {
            log::warn!(
                "{MAX_STREAMS_ENV}={raw:?} is not an integer in 1..=255; ignoring it \
                 and applying the mlrs default cap of {MLRS_MAX_STREAMS}"
            );
            None
        }
    }
}

fn install_once() -> u8 {
    // `RuntimeConfig::set` panics when the config has already been set OR read,
    // and `get` would initialize it out from under us. Going through `storage()`
    // — a public trait method — lets us do `get`'s work without either hazard.
    let storage = CubeClRuntimeConfig::storage();
    let mut slot = storage.lock();

    if let Some(existing) = slot.as_ref() {
        // Someone reached the config before the first client. Respect it (it is
        // write-once), but say so if it is the pathological default.
        let current = existing.streaming.max_streams;
        if current > MLRS_MAX_STREAMS {
            log::warn!(
                "CubeCL config was already initialized with streaming.max_streams = \
                 {current}; mlrs could not apply its cap of {MLRS_MAX_STREAMS}. Thread \
                 fan-out over mlrs estimators may exhaust device memory."
            );
        }
        return current;
    }

    // Reproduce what `RuntimeConfig::get` would have loaded, so a user's
    // `cubecl.toml` and every OTHER `CUBECL_*` env override still apply.
    let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
    let from_file = config.streaming.max_streams;

    let effective = match env_override() {
        Some(n) => n,
        // A `cubecl.toml` that names a non-default value is a deliberate choice.
        None if from_file != CUBECL_DEFAULT_MAX_STREAMS => from_file,
        None => MLRS_MAX_STREAMS,
    };

    config.streaming.max_streams = effective;
    *slot = Some(std::sync::Arc::new(config));

    if effective != from_file {
        log::info!(
            "mlrs capped CubeCL streaming.max_streams {from_file} -> {effective} \
             (one stream per OS thread, one arena per stream; mlrs serializes device \
             work behind a single pool mutex, so extra streams only fragment memory). \
             Override with {MAX_STREAMS_ENV}."
        );
    }
    effective
}
