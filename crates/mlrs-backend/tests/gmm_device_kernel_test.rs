//! `GmmDevice` KERNEL EXECUTION test (MIX-GPU) — the real, on-target-shape
//! evidence `gaussian_mixture_device_test.rs` (in `mlrs-algos`) cannot provide
//! in this sandbox.
//!
//! ## Why this file exists, and why `--features cpu` is the right backend for it
//! `gmm_device_applicable` (the EM engine's estimator-level dispatch predicate)
//! excludes the `cpu` backend and, at `f64`, excludes wgpu on this repo's own
//! dev adapter (`gmm_device.rs` module docs — the wgpu exclusion is a hard
//! SAFETY gate: `.exp()`/`.ln()` at `f64` inside a device kernel SIGSEGVs that
//! adapter's shader compiler). Both exclusions are real, but they answer
//! DIFFERENT questions than this file asks:
//!
//! - the **cpu** exclusion is a DISPATCH POLICY (`gmm_host`'s hand-written
//!   host loop already wins there — launch-per-thread overhead, not a
//!   correctness or safety concern with the kernels themselves);
//! - the **wgpu** exclusion is a genuine SAFETY gate (a shader-compiler crash).
//!
//! `cubecl-cpu` does NOT compile through a shader compiler at all — it JITs
//! the same kernel IR an `wgpu`/`cuda` backend would receive through LLVM
//! (`cubecl_cpu::compute`, one OS thread per unit —
//! [[mlrs-cubecl-cpu-execution-model]]) and evaluates `f64` `.exp()`/`.ln()`
//! natively (`capability::f64_transcendental_supported`'s own docs: "cpu /
//! cuda / rocm evaluate f64 transcendentals natively"). So `--features cpu`
//! is a SAFE, REAL execution of every kernel in `mlrs_kernels::gmm` — it is
//! not `gmm_host.rs`'s hand-written host arm, and it is not a "read the code
//! and believe it" argument.
//!
//! This file therefore constructs [`GmmDevice`] DIRECTLY, bypassing
//! `gmm_device_applicable` entirely (that predicate is a POLICY the
//! ESTIMATOR consults; the engine itself has no self-gate), and drives it
//! iteration-by-iteration against [`GmmHost`] from an IDENTICAL fixed
//! initialization, comparing `resp`/`nk`/`means`/`covariances`/`mean_lpn` at
//! EVERY iteration (not just the final one) for EACH of the four
//! `covariance_type` values — since `gmm_wlp_direct`'s branch and the M-step
//! kernel used (`gmm_cov_full_blocked` / `gmm_cov_diag_blocked` /
//! `ensure_xtx`'s once-per-fit `gmm_xtx_blocked`) differ meaningfully per
//! type.
//!
//! `cargo test -p mlrs-backend --features cpu gmm_device_kernel` on a
//! cuda/rocm-hardware host would ALSO exercise real GPU kernel execution
//! through this same file (unaffected by the wgpu-only safety gate — cuda/rocm
//! evaluate f64 transcendentals natively too); that pass is out of scope for
//! this sandbox, which has neither.
//!
//! Per AGENTS.md §2 tests live here, never in-source.

use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::gmm_device::GmmDevice;
use mlrs_backend::prims::gmm_host::{precisions_cholesky, CovarianceType, GmmHost};
use mlrs_backend::runtime::{self, ActiveRuntime};

const COV_TYPES: [CovarianceType; 4] = [
    CovarianceType::Full,
    CovarianceType::Tied,
    CovarianceType::Diag,
    CovarianceType::Spherical,
];

const N: usize = 2_000;
const D: usize = 16;
const K: usize = 4;
const REG_COVAR: f64 = 1e-6;
const N_ITERS: usize = 5;

/// Deterministic splitmix64, the same construction every other gmm test file
/// in this crate uses.
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

/// `k` well-separated isotropic blobs, `n × d` row-major — well-conditioned
/// enough that every `covariance_type`'s Cholesky stays well inside
/// `reg_covar`'s floor for all `N_ITERS`.
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

/// Tight, DELIBERATELY not loosened, agreement tolerance: both engines
/// accumulate the exact same mathematical sums (module docs on
/// `gmm_device.rs`), just in a different order (host: per-thread-chunk then
/// fold; device: per-block then fold) — the SAME class of divergence
/// `normal_eq.rs`'s host/device Gram agreement test bounds, and it bounds it
/// far tighter than the `1e-5` sklearn-oracle tolerance because there is no
/// cross-implementation slack here, only summation-order `f64` noise.
const TIGHT_TOL_ABS: f64 = 1e-8;
const TIGHT_TOL_REL: f64 = 1e-8;

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
            abs_err <= TIGHT_TOL_ABS + TIGHT_TOL_REL * e.abs(),
            "{what}: disagreement at {i}: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// A fixed, deterministic ROUND-ROBIN one-hot initial responsibility — every
/// component gets `~n/k` points and no component is ever empty, so
/// [`precisions_cholesky`] never hits [`mlrs_backend::prims::gmm_host::IllConditioned`]
/// on the first M-step regardless of `covariance_type`.
fn round_robin_resp(n: usize, k: usize) -> Vec<f64> {
    let mut resp = vec![0.0f64; n * k];
    for i in 0..n {
        resp[i * k + (i % k)] = 1.0;
    }
    resp
}

/// Drive [`GmmHost`] and [`GmmDevice`] from an IDENTICAL initialization for
/// `N_ITERS` E/M iterations, asserting agreement at EVERY iteration — not
/// just the last — for one `covariance_type`.
fn assert_engines_agree_every_iteration(ct: CovarianceType, x: &[f64]) {
    let label = ct.name();

    // --- Identical initialization for both engines (the same one
    //     `GaussianMixture::initialize`'s `RandomFromData`-shaped route would
    //     produce, minus the RNG — a fixed round-robin assignment keeps this
    //     file deterministic without needing k-means++). ---
    let resp0 = round_robin_resp(N, K);
    let mut host = GmmHost::new(x, N, D, K, ct, REG_COVAR);
    host.set_resp(&resp0);
    let (nk0, means0) = host.nk_and_means_from_resp();
    let cov0 = host.covariances(&nk0, &means0);
    let prec0 = precisions_cholesky(&cov0, K, D, ct)
        .unwrap_or_else(|e| panic!("{label}: init covariance must be well-conditioned: {e:?}"));
    let mut weights_h: Vec<f64> = nk0.iter().map(|v| v / N as f64).collect();
    let mut means_h = means0;
    let mut prec_h = prec0.clone();

    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
    let mut device = GmmDevice::new(&mut pool, x, N, D, K, ct, REG_COVAR)
        .unwrap_or_else(|e| panic!("{label}: GmmDevice::new must succeed: {e:?}"));
    let mut weights_d = weights_h.clone();
    let mut means_d = means_h.clone();
    let mut prec_d = prec_h.clone();

    for it in 0..N_ITERS {
        let (bound_h, nk_h, next_means_h) = host.e_step(&weights_h, &means_h, &prec_h);
        let cov_h = host.covariances(&nk_h, &next_means_h);
        let next_prec_h = precisions_cholesky(&cov_h, K, D, ct)
            .unwrap_or_else(|e| panic!("{label} it{it}: host covariance ill-conditioned: {e:?}"));

        let (bound_d, nk_d, next_means_d) = device
            .e_step(&mut pool, &weights_d, &means_d, &prec_d)
            .unwrap_or_else(|e| panic!("{label} it{it}: device e_step failed: {e:?}"));
        let cov_d = device
            .covariances(&mut pool, &nk_d, &next_means_d)
            .unwrap_or_else(|e| panic!("{label} it{it}: device covariances failed: {e:?}"));
        let next_prec_d = precisions_cholesky(&cov_d, K, D, ct)
            .unwrap_or_else(|e| panic!("{label} it{it}: device covariance ill-conditioned: {e:?}"));

        // --- Compare EVERY quantity the two engines exchange with the
        //     estimator's iteration loop, at THIS iteration. ---
        assert_close(
            &[bound_d],
            &[bound_h],
            &format!("{label} it{it}: mean_lpn (lower_bound_ term)"),
        );
        assert_close(&nk_d, &nk_h, &format!("{label} it{it}: nk"));
        assert_close(
            &next_means_d,
            &next_means_h,
            &format!("{label} it{it}: means"),
        );
        assert_close(&cov_d, &cov_h, &format!("{label} it{it}: covariances"));
        assert_close(
            &device.resp_to_host(&pool),
            host.resp(),
            &format!("{label} it{it}: resp"),
        );

        weights_h = nk_h.iter().map(|v| v / N as f64).collect();
        means_h = next_means_h;
        prec_h = next_prec_h;
        weights_d = nk_d.iter().map(|v| v / N as f64).collect();
        means_d = next_means_d;
        prec_d = next_prec_d;
    }

    device.release_into(&mut pool);
}

/// `GmmDevice`'s kernels agree with `GmmHost`'s hand-written host arm, for
/// EVERY `covariance_type`, at EVERY iteration of a real EM trajectory —
/// executed through `cubecl-cpu`'s LLVM JIT (see module docs for why this is a
/// genuine kernel-execution test and not a restatement of `gmm_host.rs`).
#[test]
fn device_kernels_agree_with_host_every_iteration_all_covariance_types() {
    let x = make_blobs(N, D, K, 4242);
    for ct in COV_TYPES {
        assert_engines_agree_every_iteration(ct, &x);
    }
}

/// A second, differently-seeded fixture with a different `(n, d, k)` shape —
/// catches a bug that happens to cancel out at one geometry (e.g. an
/// off-by-one in the row-block sizing, which only bites when `n` does not
/// divide the block count evenly).
#[test]
fn device_kernels_agree_with_host_at_a_second_shape() {
    const N2: usize = 777;
    const D2: usize = 4;
    const K2: usize = 2;
    let x = make_blobs(N2, D2, K2, 999);

    for ct in COV_TYPES {
        let label = ct.name();
        let resp0 = round_robin_resp(N2, K2);
        let mut host = GmmHost::new(&x, N2, D2, K2, ct, REG_COVAR);
        host.set_resp(&resp0);
        let (nk0, means0) = host.nk_and_means_from_resp();
        let cov0 = host.covariances(&nk0, &means0);
        let prec0 = precisions_cholesky(&cov0, K2, D2, ct)
            .unwrap_or_else(|e| panic!("{label}: init must be well-conditioned: {e:?}"));
        let weights0: Vec<f64> = nk0.iter().map(|v| v / N2 as f64).collect();

        let (bound_h, nk_h, means_h) = host.e_step(&weights0, &means0, &prec0);
        let cov_h = host.covariances(&nk_h, &means_h);

        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(runtime::active_client());
        let mut device = GmmDevice::new(&mut pool, &x, N2, D2, K2, ct, REG_COVAR)
            .unwrap_or_else(|e| panic!("{label}: GmmDevice::new must succeed: {e:?}"));
        let (bound_d, nk_d, means_d) = device
            .e_step(&mut pool, &weights0, &means0, &prec0)
            .unwrap_or_else(|e| panic!("{label}: device e_step failed: {e:?}"));
        let cov_d = device
            .covariances(&mut pool, &nk_d, &means_d)
            .unwrap_or_else(|e| panic!("{label}: device covariances failed: {e:?}"));

        assert_close(
            &[bound_d],
            &[bound_h],
            &format!("{label} (n=777): mean_lpn"),
        );
        assert_close(&nk_d, &nk_h, &format!("{label} (n=777): nk"));
        assert_close(&means_d, &means_h, &format!("{label} (n=777): means"));
        assert_close(&cov_d, &cov_h, &format!("{label} (n=777): covariances"));

        device.release_into(&mut pool);
    }
}
