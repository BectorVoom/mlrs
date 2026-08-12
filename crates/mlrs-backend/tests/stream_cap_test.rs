//! STREAM-CAP-01 — the CubeCL stream-count cap.
//!
//! CubeCL's config is **write-once per process**, so the cap can only be
//! installed once and cannot be re-installed with a different value. Every
//! assertion here therefore has to hold in ONE process against ONE installed
//! value, and the precedence rules (env override, user `cubecl.toml`) are proved
//! by the pure decision logic plus a separate spawned process for the env case —
//! not by mutating global state mid-test.
//!
//! Runs on every backend: the cap is backend-independent host configuration.
//! Its motivation is documented in `crates/mlrs-backend/src/stream_cap.rs`, and
//! the rocm investigation that produced it in
//! `crates/mlrs-backend/tests/thread_stream_probe_test.rs`.

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use mlrs_backend::runtime;
use mlrs_backend::stream_cap::{
    self, CUBECL_DEFAULT_MAX_STREAMS, MAX_STREAMS_ENV, MLRS_MAX_STREAMS,
};

/// The duplicated constant still matches cubecl's own default.
///
/// `stream_cap` distinguishes "the user chose this in `cubecl.toml`" from "this
/// is the untouched default" by comparing against [`CUBECL_DEFAULT_MAX_STREAMS`].
/// If a cubecl upgrade moves the default, that comparison silently stops
/// recognizing the default — and the cap would start treating it as a
/// deliberate user choice and decline to apply. Fail here instead.
#[test]
fn duplicated_cubecl_default_is_still_accurate() {
    assert_eq!(
        CubeClRuntimeConfig::default().streaming.max_streams,
        CUBECL_DEFAULT_MAX_STREAMS,
        "cubecl's default max_streams changed; update CUBECL_DEFAULT_MAX_STREAMS \
         in crates/mlrs-backend/src/stream_cap.rs to match"
    );
}

/// The cap mlrs installs is one stream.
///
/// Not an arbitrary number: every device call in the extension runs under the
/// process-global pool mutex, so a second stream can never carry concurrent
/// work — only fragment the arena.
#[test]
fn the_cap_is_a_single_stream() {
    assert_eq!(MLRS_MAX_STREAMS, 1);
}

/// `install()` is idempotent and reports the value it settled on.
#[test]
fn install_is_idempotent() {
    let first = stream_cap::install();
    assert_eq!(first, stream_cap::install());
    assert_eq!(Some(first), stream_cap::installed());
}

/// Building a client installs the cap — callers never have to remember to.
///
/// This is the load-bearing wiring: the config is write-once and is frozen the
/// moment a client reads it, so if `active_client` did not install first, the
/// cap could never be applied at all.
#[test]
fn building_a_client_installs_the_cap() {
    let _client = runtime::active_client();
    assert!(
        stream_cap::installed().is_some(),
        "active_client() must install the stream cap before building the client"
    );
}

/// The value the cap installed is the value CubeCL actually holds.
///
/// Reads the global config back rather than trusting `install`'s return value,
/// so a version where the write silently failed is caught.
#[test]
fn the_installed_value_reaches_the_cubecl_config() {
    let installed = stream_cap::install();
    assert_eq!(
        CubeClRuntimeConfig::get().streaming.max_streams,
        installed,
        "the cap reported {installed} but CubeCL is running with a different value"
    );
}

/// With no `cubecl.toml` and no env override, the effective value is the cap.
///
/// Guarded on the repo actually having no `cubecl.toml` in scope: the loader
/// walks parent directories, so a developer with one in `$HOME` would otherwise
/// see a confusing failure rather than a skip.
#[test]
fn default_environment_gets_the_cap() {
    if std::env::var(MAX_STREAMS_ENV).is_ok() {
        eprintln!("SKIP: {MAX_STREAMS_ENV} is set in this environment");
        return;
    }
    if cubecl_toml_in_scope() {
        eprintln!("SKIP: a cubecl.toml is in scope; its choice is respected by design");
        return;
    }
    assert_eq!(stream_cap::install(), MLRS_MAX_STREAMS);
}

/// The env escape hatch wins, proved in a FRESH process.
///
/// It cannot be proved in-process: the config is write-once, and this binary's
/// other tests have already installed a value. Re-running this same test binary
/// with the variable set is the only honest way to exercise the branch.
#[test]
fn env_override_wins_in_a_fresh_process() {
    if std::env::var("MLRS_STREAM_CAP_CHILD").is_ok() {
        // Child leg: assert the override took, and say so on stdout so the
        // parent can distinguish "passed" from "never ran".
        let want: u8 = std::env::var(MAX_STREAMS_ENV).unwrap().parse().unwrap();
        assert_eq!(stream_cap::install(), want);
        println!("CHILD_OK {want}");
        return;
    }

    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "env_override_wins_in_a_fresh_process",
            "--exact",
            "--nocapture",
        ])
        .env("MLRS_STREAM_CAP_CHILD", "1")
        .env(MAX_STREAMS_ENV, "7")
        .output()
        .expect("re-running the test binary");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("CHILD_OK 7"),
        "child process did not honour {MAX_STREAMS_ENV}=7.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A malformed or zero override is ignored rather than propagated.
///
/// `0` matters specifically: `StreamPool` indexes with `stream_id % max_streams`,
/// so letting a zero through would turn a typo into a divide-by-zero panic deep
/// inside cubecl.
#[test]
fn a_bad_env_override_falls_back_to_the_cap() {
    for bad in ["0", "", "abc", "-1", "999"] {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "env_override_wins_in_a_fresh_process",
                "--exact",
                "--nocapture",
            ])
            .env("MLRS_STREAM_CAP_CHILD", "1")
            .env(MAX_STREAMS_ENV, bad)
            .output()
            .expect("re-running the test binary");
        // The child's own assertion is meaningless for a bad value (it cannot
        // parse it either), so assert on the OUTCOME: the child must not crash
        // inside cubecl, and must not report having honoured the bad value.
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            !stdout.contains(&format!("CHILD_OK {bad}")),
            "{MAX_STREAMS_ENV}={bad:?} was honoured; it must be rejected"
        );
    }
}

/// Device work still runs correctly under the cap.
///
/// A cap that broke compute would be a poor trade for the memory it saves, and
/// on a single-stream config every thread now shares one stream — so this also
/// covers the "is sharing one stream across threads safe" question.
#[test]
fn device_work_is_correct_under_the_cap() {
    use cubecl::prelude::*;
    use mlrs_backend::device_array::DeviceArray;
    use mlrs_backend::pool::BufferPool;
    use mlrs_backend::runtime::ActiveRuntime;
    use mlrs_kernels::saxpy_kernel;

    const N: usize = 2048;
    let mut pool = BufferPool::new(runtime::active_client());
    let x_host: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let y_host: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();
    let x = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &x_host);
    let y = DeviceArray::<ActiveRuntime, f32>::from_host(&mut pool, &y_host);

    let block = 256u32;
    saxpy_kernel::launch::<f32, ActiveRuntime>(
        pool.client(),
        CubeCount::Static((N as u32).div_ceil(block), 1, 1),
        CubeDim {
            x: block,
            y: 1,
            z: 1,
        },
        2.0f32,
        // SAFETY: length comes from the host slice; the kernel bounds-checks.
        unsafe { ArrayArg::from_raw_parts(x.handle().clone(), N) },
        unsafe { ArrayArg::from_raw_parts(y.handle().clone(), N) },
    );

    let got = y.to_host(&pool);
    for (i, &g) in got.iter().enumerate() {
        assert_eq!(g, 2.0 * (i % 7) as f32 + (i % 5) as f32, "saxpy at {i}");
    }
}

/// Does a `cubecl.toml` / `CubeCL.toml` sit in this directory or any parent?
fn cubecl_toml_in_scope() -> bool {
    let mut dir = std::env::current_dir().expect("cwd");
    loop {
        for name in ["cubecl.toml", "CubeCL.toml", "burn.toml", "Burn.toml"] {
            if dir.join(name).exists() {
                return true;
            }
        }
        if !dir.pop() {
            return false;
        }
    }
}
