//! `BayesianGaussianMixture` DEVICE EM engine correctness tests (MIX-02-GPU).
//!
//! Mirrors `gaussian_mixture_device_test.rs` exactly, one estimator over: both
//! share [`mlrs_backend::prims::gmm_device::GmmDevice`] (the variational
//! E-step is [`GmmDevice::e_step_biased`], the plain model's is
//! [`GmmDevice::e_step`] — see `bayesian_gaussian_mixture.rs`'s module docs
//! for why they can share the whole `O(n·k·d²)` kernel set), so the same
//! `MLRS_GMM_DEVICE` `abflag` and the same `gmm_device_applicable` gate decide
//! reachability for both. Sweeps EVERY `covariance_type`, EVERY `init_params`,
//! and EVERY `weight_concentration_prior_type` string value.
//!
//! ## Why these tests SKIP (not fail) on cpu and on this environment's wgpu
//! Identical reasoning to `gaussian_mixture_device_test.rs`: `cpu` never takes
//! the device arm, and wgpu at `f64` cannot (the E-step's `logsumexp` needs
//! `f64` transcendentals, which SIGSEGV wgpu's shader compiler on this
//! sandbox's adapter). The device arm is exercised for real on cuda/rocm
//! hardware; `cargo check --features cuda`/`--features rocm` is this repo's
//! compile-cleanliness gate otherwise.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::mixture::bayesian_gaussian_mixture::BayesianGaussianMixture;
use mlrs_backend::abflag;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gmm_device::gmm_device_applicable;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::F64_TOL;

const COV_TYPES: [&str; 4] = ["full", "tied", "diag", "spherical"];
const INITS: [&str; 4] = ["kmeans", "k-means++", "random", "random_from_data"];
const PRIOR_TYPES: [&str; 2] = ["dirichlet_process", "dirichlet_distribution"];

/// Counter-based splitmix64 — the same construction
/// `gaussian_mixture_device_test.rs` uses, reproduced locally because
/// integration test binaries cannot share non-`pub` items.
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
#[allow(clippy::too_many_arguments)]
fn fit_forced<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    cov: &str,
    init: &str,
    ptype: &str,
    arm: &str,
) -> BayesianGaussianMixture<F, mlrs_algos::typestate::Fitted>
where
    F: Float + CubeElement + Pod,
{
    let _guard = abflag::force("MLRS_GMM_DEVICE", arm);
    BayesianGaussianMixture::<F>::builder()
        .n_components(K)
        .covariance_type(cov)
        .init_params(init)
        .weight_concentration_prior_type(ptype)
        .tol(1e-10)
        .max_iter(60)
        .reg_covar(1e-6)
        .random_state(Some(11))
        .build::<F>()
        .expect("valid BayesianGaussianMixture hyperparameters")
        .fit_from_host_slice(pool, x, (N, D))
        .expect("bayesian gaussian mixture fit")
}

/// Is the device arm ACTUALLY reachable on this backend, for this shape?
/// Mirrors `BayesianGaussianMixture::device_fit_applicable`'s own gate, forced
/// ON so the size floor cannot itself be the reason this reports `false` —
/// only the hard backend/capability gates can.
fn device_arm_reachable() -> bool {
    let _guard = abflag::force("MLRS_GMM_DEVICE", "1");
    gmm_device_applicable(N, D, K)
}

/// Every `covariance_type` × device-forced fit agrees with the host-forced
/// fit, on every fitted attribute that matters.
#[test]
fn device_matches_host_for_every_covariance_type() {
    if !device_arm_reachable() {
        log::warn!(
            "bayesian_mixture_device_test: skipping (device arm not reachable on this backend \
             — see module docs; expected on cpu and on this environment's wgpu)"
        );
        return;
    }
    let x = make_blobs(N, D, K, 7);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

    for cov in COV_TYPES {
        let host = fit_forced::<f64>(&mut pool, &x, cov, "kmeans", "dirichlet_process", "0");
        let dev = fit_forced::<f64>(&mut pool, &x, cov, "kmeans", "dirichlet_process", "1");

        assert_close(
            &dev.params_f64().means,
            &host.params_f64().means,
            &format!("{cov}: device vs host means_"),
        );
        assert_close(
            &dev.params_f64().covariances,
            &host.params_f64().covariances,
            &format!("{cov}: device vs host covariances_"),
        );
        assert_close(
            &dev.params_f64().weight_concentration_a,
            &host.params_f64().weight_concentration_a,
            &format!("{cov}: device vs host weight_concentration_a"),
        );
        assert_close(
            &dev.params_f64().mean_precision,
            &host.params_f64().mean_precision,
            &format!("{cov}: device vs host mean_precision_"),
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
/// engines once the E/M loop runs on device.
#[test]
fn device_matches_host_for_every_init_params() {
    if !device_arm_reachable() {
        log::warn!(
            "bayesian_mixture_device_test: skipping (device arm not reachable on this backend)"
        );
        return;
    }
    let x = make_blobs(N, D, K, 13);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

    for init in INITS {
        let host = fit_forced::<f64>(&mut pool, &x, "full", init, "dirichlet_process", "0");
        let dev = fit_forced::<f64>(&mut pool, &x, "full", init, "dirichlet_process", "1");

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

/// Every `weight_concentration_prior_type` string value agrees between
/// engines — the variational bias vector `log_weight_term` folds in a
/// different expected-log-weight term per type, so this is the one sweep that
/// actually exercises both branches of that computation on device.
#[test]
fn device_matches_host_for_every_prior_type() {
    if !device_arm_reachable() {
        log::warn!(
            "bayesian_mixture_device_test: skipping (device arm not reachable on this backend)"
        );
        return;
    }
    let x = make_blobs(N, D, K, 29);
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());

    for ptype in PRIOR_TYPES {
        let host = fit_forced::<f64>(&mut pool, &x, "full", "kmeans", ptype, "0");
        let dev = fit_forced::<f64>(&mut pool, &x, "full", "kmeans", ptype, "1");

        assert_close(
            &dev.params_f64().means,
            &host.params_f64().means,
            &format!("ptype={ptype}: device vs host means_"),
        );
        assert_close(
            &dev.params_f64().weight_concentration_a,
            &host.params_f64().weight_concentration_a,
            &format!("ptype={ptype}: device vs host weight_concentration_a"),
        );
        assert_close(
            &[dev.lower_bound()],
            &[host.lower_bound()],
            &format!("ptype={ptype}: device vs host lower_bound_"),
        );
        assert_eq!(
            dev.n_iter(),
            host.n_iter(),
            "ptype={ptype}: device vs host n_iter_"
        );
    }
}

/// `gmm_device_applicable`'s size floor actually gates something for this
/// estimator too — the predicate is shared verbatim with `GaussianMixture`
/// (both estimators' `device_fit_applicable` delegate to the same free
/// function), so this is a light sanity check rather than a duplicate of
/// `gaussian_mixture_device_test.rs`'s own coverage of the predicate itself.
#[test]
fn size_floor_keeps_tiny_shapes_on_host_by_default() {
    assert!(
        !gmm_device_applicable(4, 2, 2),
        "a 4x2 design with k=2 must never take the device EM engine by default"
    );
}
