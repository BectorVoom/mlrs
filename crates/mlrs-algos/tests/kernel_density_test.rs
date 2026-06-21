//! Plan 08-04 — KernelDensity (KERNEL-02) sklearn oracle tests.
//!
//! Activated from the 08-01 Nyquist `#[ignore]` scaffold: each function now loads
//! its committed `KernelDensity(kernel=…, bandwidth=…, atol=0, rtol=0)` fixture,
//! fits the device estimator per case, materializes `score_samples(Q)`, and
//! asserts against the sklearn FORCED-EXACT (`atol=0, rtol=0`) reference
//! log-densities within the documented KD tolerance (NOT strict 1e-5 — the
//! log-domain density has a large dynamic range, per KERNEL-02 wording).
//!
//! KernelDensity composes the v1 `distance` prim DIRECTLY (D-08 — NOT the
//! kernel-matrix prim) + a per-element KD density-value map (linear domain, exact
//! 0 out of support, D-11) + a per-query (row) log-sum-exp over the v1 `reduce`
//! prim, finalized with the host-side per-kernel `log_norm − log(N)`. It
//! implements `ScoreSamples` (length-`n` log-density, D-12), NOT a
//! `Predict`/`KNeighbors` surface.
//!
//! Case families per dtype:
//!   - **six kernels** (gaussian/tophat/epanechnikov/exponential/linear/cosine)
//!     at the fixed numeric `bandwidth = 1.0` vs `ld_<kernel>`.
//!   - **bandwidth rules** (`'scott'` / `'silverman'`, D-09): gaussian kernel with
//!     the host-resolved bandwidth vs `ld_scott` / `ld_silverman`; the resolved
//!     `bandwidth_` is cross-checked against the fixture's `bw_scott`/`bw_silverman`.
//!   - **score_samples shape** (D-12): the returned vector is length `N_QUERY`.
//!
//! Open Q1 RESOLVED here: the plain reduce-sum (no reduce-max rescale) f32 band
//! passes — the kernel values are O(1) bounded (`K(0,h)=1`), so the linear-domain
//! sum has no overflow/underflow. Rescale NOT needed.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 runs on rocm. Per AGENTS.md §2 tests live
//! in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::density::{BandwidthSpec, KdKernel, KernelDensity};
use mlrs_algos::traits::ScoreSamples;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, Tolerance};

/// KernelDensity fixture geometry (gen_oracle.py `KD_N_SAMPLES` × `KD_N_FEATURES`,
/// `KD_N_QUERY` query rows).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 3;
const N_QUERY: usize = 6;

/// The numeric bandwidth the per-kernel fixtures were generated with (gen_oracle.py
/// `bandwidth = 1.0`).
const FIXED_BANDWIDTH: f64 = 1.0;

/// The six sklearn KernelDensity kernels (D-07): the oracle array name `ld_<kernel>`
/// paired with the typed estimator kernel.
const KERNELS: [(&str, KdKernel); 6] = [
    ("ld_gaussian", KdKernel::Gaussian),
    ("ld_tophat", KdKernel::Tophat),
    ("ld_epanechnikov", KdKernel::Epanechnikov),
    ("ld_exponential", KdKernel::Exponential),
    ("ld_linear", KdKernel::Linear),
    ("ld_cosine", KdKernel::Cosine),
];

/// Documented KD log-density tolerance (NOT strict 1e-5 per KERNEL-02 wording —
/// the log-domain density has a large dynamic range). f64 uses a tight band; f32 a
/// looser one. Observed errors (cpu): gaussian/tophat/epanechnikov/exponential/
/// linear f64 ≤ ~1e-12 (host-lgamma vs Cython-lgamma parity, A1); the cosine
/// kernel's chain-rule SERIES `log_norm` (`_binary_tree.pxi.tp` 466-473) accrues
/// ~1.6e-8 — still ≫ tighter than the strict-1e-5 floor. f32 ≤ ~1e-4. The 1e-6
/// f64 band documents the cosine-series margin while staying an order of magnitude
/// inside KERNEL-02's documented contract (RESEARCH §"Claude's Discretion" / A1).
const KD_F64_BAND: Tolerance = Tolerance::new(1e-6, 1e-6);
const KD_F32_BAND: Tolerance = Tolerance::new(1e-3, 1e-3);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn host_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_density fixtures are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_density fixtures are f32/f64 only"),
    }
}

/// numpy-`allclose` element compare: pass if `|got − exp| ≤ atol + rtol·|exp|`
/// (abs-OR-rel). Mirrors `ridge_test.rs`.
fn assert_close(got: &[f64], expected: &[f64], tol: &Tolerance, what: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{what}: length mismatch got={} expected={}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        // Non-finite parity: a compact-support kernel (tophat/epanechnikov/linear/
        // cosine) returns log-density `−∞` for a query with ZERO density in its
        // support (every training point outside `bandwidth` → row_sum = 0 →
        // log(0) = −∞). sklearn returns the SAME `−∞`, so a matching infinity is a
        // PASS — but `(−∞) − (−∞) = NaN` would spuriously fail the numeric band.
        // Treat identical non-finite values (same sign of ∞) as exactly equal
        // (D-11 — the `−∞` is produced once at the terminal host `log`, matching
        // sklearn's exact-summation `−∞`).
        if !g.is_finite() || !e.is_finite() {
            assert!(
                g == e || (g.is_nan() && e.is_nan()),
                "{what}: non-finite mismatch at {i}: got={g:e} expected={e:e}"
            );
            continue;
        }
        let abs_err = (g - e).abs();
        let allclose = abs_err <= tol.abs + tol.rel * e.abs();
        assert!(
            allclose,
            "{what}: allclose failed at {i}: got={g:e} expected={e:e} \
             abs_err={abs_err:e} (atol={:e}, rtol={:e})",
            tol.abs, tol.rel
        );
    }
}

/// Fit `KernelDensity(kernel, bandwidth)` on the fixture `X` and return the host
/// `score_samples(Q)` log-density vector (length `N_QUERY`).
fn fit_score<F>(case: &OracleCase, kernel: KdKernel, bandwidth: BandwidthSpec) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case
        .expect_f64("X")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();
    let q_host: Vec<F> = case
        .expect_f64("Q")
        .iter()
        .map(|&v| f64_to::<F>(v))
        .collect();

    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let q_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &q_host);

    let mut kde = KernelDensity::<F>::new(kernel, bandwidth);
    kde.fit(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("KernelDensity::fit on a valid shape");

    let ld = kde
        .score_samples(&mut pool, &q_dev, (N_QUERY, N_FEATURES))
        .expect("score_samples after fit");
    ld.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect()
}

/// Drive the six-kernel oracle at the fixed numeric bandwidth, asserting each
/// kernel's `score_samples(Q)` against the fixture `ld_<kernel>` within `tol`.
fn run_six_kernels<F>(case: &OracleCase, tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    for (name, kernel) in KERNELS {
        let expected = case.expect_f64(name);
        assert_eq!(expected.len(), N_QUERY, "fixture {name} is length n_query");
        let got = fit_score::<F>(case, kernel, BandwidthSpec::Numeric(FIXED_BANDWIDTH));
        assert_close(&got, &expected, tol, &format!("{label} {name}"));
    }
}

/// Drive the scott/silverman bandwidth-rule oracle (gaussian kernel, D-09):
/// cross-check the host-resolved `bandwidth_` against the fixture, then assert
/// `score_samples(Q)` against `ld_scott` / `ld_silverman`.
fn run_bandwidth_rules<F>(case: &OracleCase, tol: &Tolerance, bw_tol: &Tolerance, label: &str)
where
    F: Float + CubeElement + Pod,
{
    let cases = [
        ("ld_scott", "bw_scott", BandwidthSpec::Scott),
        ("ld_silverman", "bw_silverman", BandwidthSpec::Silverman),
    ];
    for (ld_name, bw_name, spec) in cases {
        // Cross-check the resolved bandwidth_ against sklearn's bandwidth_.
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_host: Vec<F> = case
            .expect_f64("X")
            .iter()
            .map(|&v| f64_to::<F>(v))
            .collect();
        let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
        let mut kde = KernelDensity::<F>::new(KdKernel::Gaussian, spec);
        kde.fit(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
            .expect("KernelDensity::fit (bandwidth rule)");
        let resolved = kde.bandwidth().expect("bandwidth_ after fit");
        let expected_bw = case.expect_f64(bw_name)[0];
        assert_close(
            &[resolved],
            &[expected_bw],
            bw_tol,
            &format!("{label} {bw_name} resolution"),
        );

        // Assert the log-densities for the resolved-bandwidth gaussian KDE.
        let expected = case.expect_f64(ld_name);
        let got = fit_score::<F>(case, KdKernel::Gaussian, spec);
        assert_close(&got, &expected, tol, &format!("{label} {ld_name}"));
    }
}

/// KERNEL-02 six-kernel log-densities vs sklearn forced-exact, f32 (`KD_F32_BAND`).
/// Runs on every backend (the f32 gate is rocm; cpu also exercises f32).
#[test]
fn kernel_density_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kernel_density_f32_seed42.npz")).expect("load kernel_density_f32");
    run_six_kernels::<f32>(&case, &KD_F32_BAND, "kernel_density f32");
}

/// KERNEL-02 six-kernel log-densities vs sklearn forced-exact, f64 (`KD_F64_BAND`).
/// Gated by `skip_f64_with_log` (cpu runs; rocm skips).
#[test]
fn kernel_density_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_density f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let case = load_npz(fixture("kernel_density_f64_seed42.npz")).expect("load kernel_density_f64");
    run_six_kernels::<f64>(&case, &KD_F64_BAND, "kernel_density f64");
}

/// KERNEL-02 bandwidth-rule (scott/silverman, D-09) log-densities + resolved-`bandwidth_`
/// cross-check vs sklearn, f64. Gated by `skip_f64_with_log` (cpu runs; rocm skips).
#[test]
fn kernel_density_bandwidth_rules_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_density bw f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("kernel_density_f64_seed42.npz")).expect("load kernel_density_f64");
    run_bandwidth_rules::<f64>(&case, &KD_F64_BAND, &KD_F64_BAND, "kernel_density bw f64");
}

/// KERNEL-02 bandwidth-rule (scott/silverman) log-densities + resolved-`bandwidth_`
/// cross-check vs sklearn, f32. Runs on every backend.
#[test]
fn kernel_density_bandwidth_rules_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kernel_density_f32_seed42.npz")).expect("load kernel_density_f32");
    run_bandwidth_rules::<f32>(&case, &KD_F32_BAND, &KD_F32_BAND, "kernel_density bw f32");
}

/// ScoreSamples<F> length-n contract (D-12, the VALIDATION.md `score_samples`
/// Nyquist case): `score_samples(Q)` returns a length-`N_QUERY` (= rows of Q)
/// vector, NOT a `Predict`/reconstruction shape. f32 path (runs on every backend).
#[test]
fn score_samples_shape_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case = load_npz(fixture("kernel_density_f32_seed42.npz")).expect("load kernel_density_f32");
    let got = fit_score::<f32>(&case, KdKernel::Gaussian, BandwidthSpec::Numeric(FIXED_BANDWIDTH));
    assert_eq!(
        got.len(),
        N_QUERY,
        "score_samples returns a length-n_query log-density vector (D-12)"
    );
}
