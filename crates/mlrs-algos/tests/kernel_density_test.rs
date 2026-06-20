//! Plan 08-04 — KernelDensity (KERNEL-02) sklearn oracle tests.
//!
//! **Wave-0 Nyquist scaffold (08-01).** Every function below is `#[ignore]`d and
//! asserts ONLY that its committed `kernel_density_*` oracle fixture loads and is
//! shape-well-formed (the `X` training matrix, the `Q` query matrix, the six
//! per-kernel reference log-densities `ld_gaussian`/`ld_tophat`/`ld_epanechnikov`/
//! `ld_exponential`/`ld_linear`/`ld_cosine`, and the `ld_scott`/`ld_silverman`
//! bandwidth-rule cases). It does NOT reference the not-yet-written
//! `KernelDensity` estimator nor the `ScoreSamples` call beyond the trait the
//! 08-01 scaffold already exposes, so this test crate COMPILES today. The Wave-2
//! plan (08-04) removes `#[ignore]`, fits the device `KernelDensity` per case,
//! materializes `score_samples(Q)`, and asserts against the sklearn
//! forced-exact (`atol=0, rtol=0`) reference within the documented KD tolerance
//! (NOT strict 1e-5 — large-dynamic-range log-density, per KERNEL-02 wording).
//!
//! KernelDensity stores the fitted `X_fit_` + resolved `bandwidth` (numeric or
//! scott/silverman host closed-form, D-09) and computes per-query log-density via
//! `distance` + a per-element KD kernel-value map + a per-query log-sum-exp over
//! the v1 `reduce` prim (D-08/D-10/D-11). It implements `ScoreSamples`, NOT a
//! `Predict`/`KNeighbors` surface (D-12 — it is NOT a neighbor estimator).
//!
//! Case families per dtype: all six kernels at a fixed numeric bandwidth, plus a
//! `'scott'` and a `'silverman'` bandwidth-rule case (D-09).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate verbatim (cpu runs
//! f64; rocm skips-with-log, D-07). f32 runs on rocm. Per AGENTS.md §2 tests live
//! in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use mlrs_backend::capability;
use mlrs_core::{load_npz, OracleCase, Tolerance};

/// KernelDensity fixture geometry (gen_oracle.py `KD_N_SAMPLES` × `KD_N_FEATURES`,
/// `KD_N_QUERY` query rows).
const N_SAMPLES: usize = 10;
const N_FEATURES: usize = 3;
const N_QUERY: usize = 6;

/// The six sklearn KernelDensity kernels (D-10) carried as the per-kernel oracle
/// array names `ld_<kernel>`.
const KERNELS: [&str; 6] = [
    "ld_gaussian",
    "ld_tophat",
    "ld_epanechnikov",
    "ld_exponential",
    "ld_linear",
    "ld_cosine",
];

/// Documented KD log-density tolerance (NOT strict 1e-5 per KERNEL-02 wording —
/// the log-domain density has a large dynamic range). f64 uses a tight band; f32
/// a looser one. Set FROM the measurement printed by the Wave-2 value test.
#[allow(dead_code)]
const KD_F64_BAND: Tolerance = Tolerance::new(1e-5, 1e-5);
#[allow(dead_code)]
const KD_F32_BAND: Tolerance = Tolerance::new(1e-4, 1e-3);

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// Load a kernel_density oracle blob and assert the inputs + all per-kernel and
/// bandwidth-rule reference log-densities are shape-well-formed (the Wave-0
/// contract: fixture-load + shape only, no compute).
fn assert_fixture_well_formed(name: &str) -> OracleCase {
    let case = load_npz(fixture(name)).expect("kernel_density fixture loads");
    assert_eq!(
        case.expect_f64("X").len(),
        N_SAMPLES * N_FEATURES,
        "X is n_samples × n_features"
    );
    assert_eq!(
        case.expect_f64("Q").len(),
        N_QUERY * N_FEATURES,
        "Q is n_query × n_features"
    );
    for k in KERNELS {
        assert_eq!(case.expect_f64(k).len(), N_QUERY, "{k} is length n_query");
    }
    for k in ["ld_scott", "ld_silverman"] {
        assert_eq!(case.expect_f64(k).len(), N_QUERY, "{k} is length n_query");
    }
    case
}

/// KERNEL-02 log-densities vs sklearn `KernelDensity.score_samples` for all six
/// kernels (forced-exact `atol=0, rtol=0`), f64 at the documented `KD_F64_BAND`.
/// Gated by `skip_f64_with_log` (cpu runs; rocm skips). Wave-2 (08-04) removes
/// `#[ignore]` and fits the device estimator via `ScoreSamples`.
#[test]
#[ignore = "Wave-0 scaffold: KernelDensity estimator lands in plan 08-04"]
fn kernel_density_all_kernels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_density f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    let _ = (assert_fixture_well_formed("kernel_density_f64_seed42.npz"), &KD_F64_BAND);
}

/// KERNEL-02 log-densities vs sklearn at the documented f32 band (`KD_F32_BAND`).
/// Runs on every backend (the f32 gate is rocm; cpu also exercises f32). Wave-2
/// (08-04) removes `#[ignore]` and fits the device estimator.
#[test]
#[ignore = "Wave-0 scaffold: KernelDensity estimator lands in plan 08-04"]
fn kernel_density_all_kernels_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let _ = assert_fixture_well_formed("kernel_density_f32_seed42.npz");
}

/// KERNEL-02 bandwidth-rule (scott/silverman, D-09) log-densities vs sklearn,
/// f64. Verifies the host bandwidth-resolution closed form. Wave-2 (08-04)
/// removes `#[ignore]` and fits the device estimator with the string rule.
#[test]
#[ignore = "Wave-0 scaffold: KernelDensity bandwidth rules land in plan 08-04"]
fn kernel_density_bandwidth_rules_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("kernel_density bw f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    let case = assert_fixture_well_formed("kernel_density_f64_seed42.npz");
    let _ = (
        case.expect_f64("ld_scott"),
        case.expect_f64("ld_silverman"),
        case.expect_f64("bw_scott"),
        case.expect_f64("bw_silverman"),
    );
}
