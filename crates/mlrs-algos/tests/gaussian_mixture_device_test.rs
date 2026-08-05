//! `GaussianMixture` DEVICE EM engine correctness tests (MIX-GPU).
//!
//! [`mlrs_backend::prims::gmm_device`] adds a genuine on-device EM engine for
//! large `n` on cuda/rocm; these tests force it via the `MLRS_GMM_DEVICE`
//! `abflag` override (`"1"` prefers device, subject to the HARD capability
//! gates `gmm_device_applicable` checks first — backend, `f64` device kernels,
//! `f64` transcendentals) and assert it reproduces the SAME fit
//! [`mlrs_backend::prims::gmm_host::GmmHost`] does, for EVERY
//! `covariance_type` and EVERY `init_params` string value.
//!
//! ## Why these tests SKIP (not fail) on cpu and on this environment's wgpu
//! `gmm_device_applicable`'s gates are hard correctness/safety gates, not a
//! preference the `MLRS_GMM_DEVICE` override can push through:
//!
//! - **cpu** never takes the device arm (`gmm_device.rs` module docs — the
//!   architectural reason from `gmm_host`, launch overhead, still applies).
//! - **wgpu at `f64`** never takes it EITHER, because
//!   [`mlrs_backend::capability::f64_transcendental_supported`] is a hard
//!   `false` there: this estimator's E-step evaluates `.exp()`/`.ln()` at
//!   `f64` inside a device kernel, and on the wgpu adapter in this repo's own
//!   dev environment that SEGFAULTS the shader compiler (documented on that
//!   function, and hit for real by `prims::lbfgs::softmax_loss_grad`'s
//!   identical shape). Forcing past that gate in a test would risk crashing
//!   the whole test binary for a wgpu run, which is a strictly worse outcome
//!   than skipping.
//!
//! So on cpu/wgpu — the only backends available in this sandbox — every test
//! below detects `gmm_device_applicable() == false` up front and returns early
//! with a logged reason, exactly like the `capability::skip_f64_with_log`
//! convention used throughout this crate for backend-gated coverage. The
//! device arm is exercised for real only on cuda/rocm hardware; `cargo check
//! --features cuda`/`--features rocm` is this repo's compile-cleanliness gate
//! for those backends in this environment (AGENTS.md / project constraints).
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::mixture::gaussian_mixture::GaussianMixture;
use mlrs_backend::abflag;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gmm_device::gmm_device_applicable;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::F64_TOL;

const COV_TYPES: [&str; 4] = ["full", "tied", "diag", "spherical"];
const INITS: [&str; 4] = ["kmeans", "k-means++", "random", "random_from_data"];

/// Counter-based splitmix64 — the same construction used by
/// `gaussian_mixture_perf_test.rs` / `bench_gmm.py`, reproduced locally so
/// this file has no cross-test-binary dependency (integration test binaries
/// cannot share non-`pub` items).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// `k` well-separated isotropic blobs, `n × d` row-major — large enough
/// (`n·k·d` product) to trip `gmm_device_applicable`'s default size floor even
/// WITHOUT the `MLRS_GMM_DEVICE` override, on a backend where the hard gates
/// hold.
fn make_blobs(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut centers = vec![0.0f64; k * d];
    for slot in centers.iter_mut() {
        *slot = (uniform01(&mut s) * 2.0 - 1.0) * 10.0;
    }
    let mut x = vec![0.0f64; n * d];
    for i in 0..n {
        let c = i % k;
        for j in 0..d {
            let u1 = uniform01(&mut s).max(f64::MIN_POSITIVE);
            let u2 = uniform01(&mut s);
            let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            x[i * d + j] = centers[c * d + j] + g;
        }
    }
    x
}

const N: usize = 6_000;
const D: usize = 8;
const K: usize = 4;

/// numpy-`allclose` element compare: `|got − exp| ≤ atol + rtol·|exp|`.
fn assert_close(got: &[f64], expected: &[f64], what: &str) {
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
            abs_err <= F64_TOL.abs + F64_TOL.rel * e.abs(),
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// Fit once with `MLRS_GMM_DEVICE` forced to `arm` (`"0"` host, `"1"` device —
/// subject to the hard gates `gmm_device_applicable` checks first).
fn fit_forced<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    cov: &str,
    init: &str,
    arm: &str,
) -> GaussianMixture<F, mlrs_algos::typestate::Fitted>
where
    F: Float + CubeElement + Pod,
{
    let _guard = abflag::force("MLRS_GMM_DEVICE", arm);
    GaussianMixture::<F>::builder()
        .n_components(K)
        .covariance_type(cov)
        .init_params(init)
        .tol(1e-10)
        .max_iter(60)
        .reg_covar(1e-6)
        .random_state(Some(11))
        .build::<F>()
        .expect("valid GaussianMixture hyperparameters")
        .fit_from_host_slice(pool, x, (N, D))
        .expect("gaussian mixture fit")
}

/// Is the device arm ACTUALLY reachable on this backend, for this shape? Mirrors
/// `GaussianMixture::device_fit_applicable`'s own gate, forced ON so the size
/// floor cannot itself be the reason this reports `false` — only the hard
/// backend/capability gates can.
fn device_arm_reachable() -> bool {
    let _guard = abflag::force("MLRS_GMM_DEVICE", "1");
    gmm_device_applicable(N, D, K)
}

/// Every `covariance_type` × device-forced fit agrees with the host-forced fit,
/// on every fitted attribute that matters (means_, weights_, covariances_,
/// lower_bound_, n_iter_, and the training labels_).
#[test]
fn device_matches_host_for_every_covariance_type() {
    if !device_arm_reachable() {
        log::warn!(
            "gaussian_mixture_device_test: skipping (device arm not reachable on this backend \
             — see module docs; expected on cpu and on this environment's wgpu)"
        );
        return;
    }
    let x = make_blobs(N, D, K, 7);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

    for cov in COV_TYPES {
        let host = fit_forced::<f64>(&mut pool, &x, cov, "kmeans", "0");
        let dev = fit_forced::<f64>(&mut pool, &x, cov, "kmeans", "1");

        assert_close(
            &dev.params_f64().means,
            &host.params_f64().means,
            &format!("{cov}: device vs host means_"),
        );
        assert_close(
            &dev.params_f64().weights,
            &host.params_f64().weights,
            &format!("{cov}: device vs host weights_"),
        );
        assert_close(
            &dev.params_f64().covariances,
            &host.params_f64().covariances,
            &format!("{cov}: device vs host covariances_"),
        );
        assert_close(
            &[dev.lower_bound()],
            &[host.lower_bound()],
            &format!("{cov}: device vs host lower_bound_"),
        );
        assert_eq!(
            dev.n_iter(),
            host.n_iter(),
            "{cov}: device vs host n_iter_ (device={}, host={})",
            dev.n_iter(),
            host.n_iter()
        );
        assert_eq!(
            dev.labels(),
            host.labels(),
            "{cov}: device vs host training labels_ diverge"
        );
    }
}

/// Every `init_params` string value still produces a fit that agrees between
/// engines once the E/M loop runs on device — a smaller test than the
/// `covariance_type` sweep (initialization itself always stays host-only, per
/// the design, so this exercises "device E/M-step correctly continues from
/// every init route's starting parameters", not the init routes themselves).
#[test]
fn device_matches_host_for_every_init_params() {
    if !device_arm_reachable() {
        log::warn!(
            "gaussian_mixture_device_test: skipping (device arm not reachable on this backend)"
        );
        return;
    }
    let x = make_blobs(N, D, K, 13);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

    for init in INITS {
        let host = fit_forced::<f64>(&mut pool, &x, "full", init, "0");
        let dev = fit_forced::<f64>(&mut pool, &x, "full", init, "1");

        assert_close(
            &dev.params_f64().means,
            &host.params_f64().means,
            &format!("init={init}: device vs host means_"),
        );
        assert_close(
            &[dev.lower_bound()],
            &[host.lower_bound()],
            &format!("init={init}: device vs host lower_bound_"),
        );
        assert_eq!(
            dev.n_iter(),
            host.n_iter(),
            "init={init}: device vs host n_iter_"
        );
    }
}

/// `gmm_device_applicable`'s size floor actually gates something: a TINY shape
/// (well below the `2^20` `n·k·d` floor) stays on the host arm even when the
/// override is absent, which this checks directly at the prim level (no fit
/// needed) so it runs identically on every backend, including cpu, where the
/// backend gate ALSO holds — both reasons independently forbid the device arm
/// here, and the point of this test is only that the predicate is not
/// vacuously `true`.
#[test]
fn size_floor_keeps_tiny_shapes_on_host_by_default() {
    assert!(
        !gmm_device_applicable(4, 2, 2),
        "a 4x2 design with k=2 must never take the device EM engine by default"
    );
}
