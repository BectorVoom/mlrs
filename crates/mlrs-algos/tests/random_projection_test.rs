//! Plan 07-06 — RandomProjection (PROJ-01/02) property + value-oracle tests.
//!
//! The gate here is a STRUCTURAL PROPERTY SET, **NOT** the 1e-5 oracle (D-12).
//! The RNG is host SplitMix64, not numpy's MT19937, so the projection matrix
//! CANNOT match sklearn element-wise; only [`johnson_lindenstrauss_min_dim`] is
//! value-matched (`random_projection_jl_min_dim`). The property gate covers:
//! the JL distortion bound (`(1−eps)·‖u−v‖² ≤ ‖proj(u)−proj(v)‖² ≤
//! (1+eps)·‖u−v‖²`) AVERAGED over `JL_TRIALS`, the matrix moment stats
//! (Gaussian mean≈0 / var≈1/n_components; Achlioptas density + ±v values),
//! seed-reproducibility (same `u64` seed → identical matrix), and the
//! `transform == X·componentsᵀ` self-consistency (no centering — D-12).
//!
//! D-11 mitigates strict-band JL flakiness with a FIXED SplitMix64 seed (per
//! trial) + averaging over `JL_TRIALS`: each trial uses a distinct fixed seed so
//! the averaged statistic concentrates, and every run/backend draws the IDENTICAL
//! matrices (bit-reproducible — never `OsRng`).
//!
//! Per AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::projection::gaussian::{
    johnson_lindenstrauss_min_dim, GaussianRandomProjection, NComponents,
};
use mlrs_algos::projection::sparse::SparseRandomProjection;
// Phase 16 (D-01): BOTH random-projection estimators are now on the typestate
// surface (consuming-self `Fit`/`Transform`, builder construction); the legacy
// `crate::traits` glob is gone (projection module complete). The typestate traits
// are imported under disambiguating `Typestate*` aliases and called via UFCS.
use mlrs_algos::typestate::{Fit as TypestateFit, Transform as TypestateTransform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::load_npz;

/// PINNED averaging trial count for the JL distortion / moment property gates
/// (D-11 — a single unlucky draw never flips the strict band; each trial uses a
/// distinct FIXED seed so the run stays bit-reproducible across runs/backends).
const JL_TRIALS: usize = 50;

/// johnson_lindenstrauss_min_dim grid sizes (gen_oracle.py `JL_N_SAMPLES` /
/// `JL_EPS` are length-3 each → a 3×3 `min_dim` matrix).
const JL_GRID: usize = 3;

/// Property-gate projection geometry: a moderate-dimensional source projected to
/// a smaller embedding so the JL distortion band is meaningful and the moment
/// statistics have enough entries to concentrate.
const PROP_SAMPLES: usize = 40;
const PROP_FEATURES: usize = 64;
const PROP_COMPONENTS: usize = 32;
/// Base seed; each trial uses `BASE_SEED + trial` (a distinct FIXED seed).
const BASE_SEED: u64 = 0x07_06_0000_0000;

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
        _ => unreachable!("projection tests are f32/f64 only"),
    }
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("projection tests are f32/f64 only"),
    }
}

/// Deterministic pseudo-random source matrix `X` (PROP_SAMPLES × PROP_FEATURES)
/// built from a fixed SplitMix64-style stream — the DATA being projected (the
/// projection MATRIX comes from the estimator's own seeded RNG). Reusing a
/// simple host stream keeps the test data bit-reproducible across backends.
fn data_matrix<F: Pod>(seed: u64, rows: usize, cols: usize) -> Vec<F> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let u = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
        // map [0,1) → [-1, 1)
        2.0 * u - 1.0
    };
    (0..rows * cols).map(|_| f64_to::<F>(next())).collect()
}

/// `johnson_lindenstrauss_min_dim` VALUE oracle (the ONE value-matched RP check,
/// D-12): the integer min-dim grid matches sklearn over the `(n_samples, eps)`
/// grid (`min_dim[i*JL_GRID + j] = jl_min_dim(n_samples[i], eps[j])`).
#[test]
fn random_projection_jl_min_dim() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let case =
        load_npz(fixture("jl_min_dim_f32_seed42.npz")).expect("load jl_min_dim_f32");

    let n_samples = case.expect_f64("n_samples");
    let eps = case.expect_f64("eps");
    let min_dim = case.expect_f64("min_dim");
    assert_eq!(n_samples.len(), JL_GRID, "n_samples grid len");
    assert_eq!(eps.len(), JL_GRID, "eps grid len");
    assert_eq!(
        case.shape("min_dim").expect("min_dim").to_vec(),
        vec![JL_GRID as u64, JL_GRID as u64],
        "min_dim is a JL_GRID×JL_GRID matrix"
    );

    for (i, &n) in n_samples.iter().enumerate() {
        for (j, &e) in eps.iter().enumerate() {
            assert!((0.0..1.0).contains(&e), "eps {e} must be in (0, 1)");
            let got = johnson_lindenstrauss_min_dim(n, e)
                .expect("eps in (0,1) → Ok");
            let expected = min_dim[i * JL_GRID + j] as usize;
            // VALUE-matched: integer-exact (well within 1e-5).
            assert_eq!(
                got, expected,
                "jl_min_dim(n={n}, eps={e}) = {got}, oracle = {expected}"
            );
        }
    }

    // eps ∉ (0, 1) is rejected (ASVS V5 / T-07-10).
    assert!(johnson_lindenstrauss_min_dim(100.0, 0.0).is_err());
    assert!(johnson_lindenstrauss_min_dim(100.0, 1.0).is_err());
    assert!(johnson_lindenstrauss_min_dim(100.0, -0.1).is_err());
    assert!(johnson_lindenstrauss_min_dim(100.0, f64::NAN).is_err());
}

/// Gaussian projection-matrix moment stats (PROPERTY, NOT 1e-5 — D-12): the
/// `N(0, 1/n_components)` `components_` has mean ≈ 0 and variance ≈
/// `1/n_components`, AVERAGED over `JL_TRIALS` distinct-but-fixed seeds (D-11)
/// so the strict band concentrates.
#[test]
fn random_projection_gaussian_moments() {
    if capability::skip_f64_with_log() {
        // f64-capable backends run the f64 stat; otherwise still run f32 below.
    }
    run_gaussian_moments::<f32>();
    if !capability::skip_f64_with_log() {
        run_gaussian_moments::<f64>();
    }
}

fn run_gaussian_moments<F>()
where
    F: Float + CubeElement + Pod,
{
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(1, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);

    let mut mean_acc = 0.0_f64;
    let mut var_acc = 0.0_f64;
    for trial in 0..JL_TRIALS {
        let rp = GaussianRandomProjection::<F>::builder()
            .n_components(NComponents::Fixed(PROP_COMPONENTS))
            .seed(BASE_SEED + trial as u64)
            .eps(0.1)
            .build::<F>()
            .expect("gaussian build");
        let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
            .expect("gaussian fit");
        let comp: Vec<f64> = rp
            .components(&pool)
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let n = comp.len() as f64;
        let mean = comp.iter().sum::<f64>() / n;
        let var = comp.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
        mean_acc += mean;
        var_acc += var;
    }
    let mean = mean_acc / JL_TRIALS as f64;
    let var = var_acc / JL_TRIALS as f64;
    let target_var = 1.0 / PROP_COMPONENTS as f64;

    // Strict bands (D-10), reproducible via averaging (D-11). Mean ≈ 0; var
    // within 5% of 1/n_components averaged over the trials.
    assert!(
        mean.abs() < 2e-3,
        "averaged Gaussian components_ mean {mean} not ≈ 0"
    );
    assert!(
        (var - target_var).abs() / target_var < 0.05,
        "averaged Gaussian components_ var {var} not ≈ 1/n_components {target_var}"
    );
}

/// `transform == X · components_ᵀ` self-consistency (PROPERTY, exact to GEMM
/// tolerance): the device transform equals a host-recomputed `X · components_ᵀ`
/// (no centering — RandomProjection does not center, D-12).
#[test]
fn random_projection_gaussian_self_consistency() {
    run_gaussian_self_consistency::<f32>();
    if !capability::skip_f64_with_log() {
        run_gaussian_self_consistency::<f64>();
    }
}

fn run_gaussian_self_consistency<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(7, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);

    let rp = GaussianRandomProjection::<F>::builder()
        .n_components(NComponents::Fixed(PROP_COMPONENTS))
        .seed(BASE_SEED)
        .eps(0.1)
        .build::<F>()
        .expect("gaussian build");
    let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
        .expect("gaussian fit");
    let nc = rp.n_components_();
    assert_eq!(nc, PROP_COMPONENTS, "Fixed n_components resolves verbatim");

    let z = TypestateTransform::transform(&rp, &mut pool, &x_dev, (PROP_SAMPLES, PROP_FEATURES))
        .expect("transform");
    let z_host: Vec<f64> = z.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();

    let comp: Vec<f64> = rp
        .components(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let x64: Vec<f64> = x_host.iter().map(|&v| host_to_f64(v)).collect();

    // Z[r][c] = sum_f X[r][f] * components_[c][f]  (componentsᵀ).
    for r in 0..PROP_SAMPLES {
        for c in 0..nc {
            let mut acc = 0.0_f64;
            for f in 0..PROP_FEATURES {
                acc += x64[r * PROP_FEATURES + f] * comp[c * PROP_FEATURES + f];
            }
            let got = z_host[r * nc + c];
            let tol = 1e-4 + 1e-4 * acc.abs();
            assert!(
                (got - acc).abs() <= tol,
                "self-consistency Z[{r}][{c}] got={got} expected={acc}"
            );
        }
    }
}

/// JL distortion bound (PROPERTY, NOT 1e-5 — D-12): pairwise squared distances
/// are preserved within `(1 ± eps)` on AVERAGE over `JL_TRIALS` fixed-seed draws
/// (D-11 — averaging makes the strict band reproducible, not seed-fragile).
#[test]
fn random_projection_gaussian_jl_distortion() {
    run_jl_distortion::<f32>(false);
    if !capability::skip_f64_with_log() {
        run_jl_distortion::<f64>(false);
    }
}

/// Run the averaged JL-distortion property over `JL_TRIALS`. `sparse=false`
/// drives the Gaussian estimator; `sparse=true` drives the Achlioptas estimator
/// (shared property discipline — D-11).
fn run_jl_distortion<F>(sparse: bool)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(13, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);
    let x64: Vec<f64> = x_host.iter().map(|&v| host_to_f64(v)).collect();

    // Sample pairs (a fixed deterministic set) whose distortion we average.
    let pairs: Vec<(usize, usize)> = (0..PROP_SAMPLES)
        .flat_map(|i| ((i + 1)..PROP_SAMPLES).map(move |j| (i, j)))
        .collect();

    let mut mean_ratio = 0.0_f64;
    let mut count = 0usize;
    for trial in 0..JL_TRIALS {
        let seed = BASE_SEED + trial as u64;
        let z_host: Vec<f64> = if sparse {
            let rp = SparseRandomProjection::<F>::builder()
                .n_components(NComponents::Fixed(PROP_COMPONENTS))
                .seed(seed)
                .eps(0.1)
                .density(None)
                .build::<F>()
                .expect("sparse build");
            let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
                .expect("sparse fit");
            TypestateTransform::transform(&rp, &mut pool, &x_dev, (PROP_SAMPLES, PROP_FEATURES))
                .expect("transform")
                .to_host(&pool)
                .iter()
                .map(|&v| host_to_f64(v))
                .collect()
        } else {
            let rp = GaussianRandomProjection::<F>::builder()
                .n_components(NComponents::Fixed(PROP_COMPONENTS))
                .seed(seed)
                .eps(0.1)
                .build::<F>()
                .expect("gaussian build");
            let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
                .expect("gaussian fit");
            TypestateTransform::transform(&rp, &mut pool, &x_dev, (PROP_SAMPLES, PROP_FEATURES))
                .expect("transform")
                .to_host(&pool)
                .iter()
                .map(|&v| host_to_f64(v))
                .collect()
        };

        for &(i, j) in &pairs {
            let mut d_src = 0.0_f64;
            for f in 0..PROP_FEATURES {
                let d = x64[i * PROP_FEATURES + f] - x64[j * PROP_FEATURES + f];
                d_src += d * d;
            }
            let mut d_proj = 0.0_f64;
            for c in 0..PROP_COMPONENTS {
                let d = z_host[i * PROP_COMPONENTS + c] - z_host[j * PROP_COMPONENTS + c];
                d_proj += d * d;
            }
            if d_src > 1e-9 {
                mean_ratio += d_proj / d_src;
                count += 1;
            }
        }
    }
    let mean_ratio = mean_ratio / count as f64;
    // The JL embedding is an isometry IN EXPECTATION: the averaged
    // projected/source squared-distance ratio concentrates at 1 (D-11
    // averaging). Strict band: within 5% of 1.0.
    assert!(
        (mean_ratio - 1.0).abs() < 0.05,
        "averaged JL distortion ratio {mean_ratio} not ≈ 1 (sparse={sparse})"
    );
}

/// Seed reproducibility (PROPERTY / T-07-02): the SAME `u64` seed → an identical
/// projection matrix; a DIFFERENT seed differs (host SplitMix64 is deterministic,
/// never `OsRng`). The cpu==rocm cross-backend identity is the phase gate (the
/// host PRNG stream is backend-independent).
#[test]
fn random_projection_seed_reproducible() {
    run_seed_reproducible::<f32>();
    if !capability::skip_f64_with_log() {
        run_seed_reproducible::<f64>();
    }
}

fn run_seed_reproducible<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(1, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);

    let fit_components = |pool: &mut BufferPool<ActiveRuntime>, seed: u64| -> Vec<F> {
        let rp = GaussianRandomProjection::<F>::builder()
            .n_components(NComponents::Fixed(PROP_COMPONENTS))
            .seed(seed)
            .eps(0.1)
            .build::<F>()
            .expect("gaussian build");
        let rp = TypestateFit::fit(rp, pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
            .expect("gaussian fit");
        rp.components(pool)
    };

    let a = fit_components(&mut pool, BASE_SEED);
    let b = fit_components(&mut pool, BASE_SEED);
    let c = fit_components(&mut pool, BASE_SEED + 1);

    // Same seed → BYTE-identical matrix (the T-07-02 hard gate guaranteeing
    // cpu==rocm). Different seed → differs.
    let a64: Vec<f64> = a.iter().map(|&v| host_to_f64(v)).collect();
    let b64: Vec<f64> = b.iter().map(|&v| host_to_f64(v)).collect();
    let c64: Vec<f64> = c.iter().map(|&v| host_to_f64(v)).collect();
    assert_eq!(a64, b64, "same seed → identical components_");
    assert_ne!(a64, c64, "different seed → different components_");

    // 'auto' n_components resolves via JL and is reproducible too.
    let auto = GaussianRandomProjection::<F>::builder()
        .n_components(NComponents::Auto)
        .seed(BASE_SEED)
        .eps(0.5)
        .build::<F>()
        .expect("auto build");
    let auto = TypestateFit::fit(auto, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
        .expect("auto fit");
    let expected_nc = johnson_lindenstrauss_min_dim(PROP_SAMPLES as f64, 0.5)
        .expect("jl");
    assert_eq!(
        auto.n_components_(),
        expected_nc,
        "Auto n_components resolves via johnson_lindenstrauss_min_dim"
    );
}

/// BLDR-01: the zero-arg `new()` and the `builder().build()` default agree on
/// every hyperparameter (the single-source-of-defaults contract, D-08).
#[test]
fn gaussian_random_projection_defaults_equal() {
    let from_new = GaussianRandomProjection::<f64>::new();
    let from_builder = GaussianRandomProjection::<f64>::builder()
        .build::<f64>()
        .expect("default build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "GaussianRandomProjection::new() and builder().build() must share defaults"
    );
}

// ===========================================================================
// SparseRandomProjection (Achlioptas, dense storage) — Task 2.
// ===========================================================================

/// Sparse Achlioptas matrix moment stats (PROPERTY, NOT 1e-5 — D-12): the
/// generated `components_` nonzero fraction ≈ `density` and every nonzero value
/// is exactly `±v = ±sqrt((1/density)/n_components)`, AVERAGED over `JL_TRIALS`
/// fixed seeds (D-11). `components_` are stored DENSE (D-12); sparse INPUT
/// densification happens at the Python ingress (Plan 07).
#[test]
fn random_projection_sparse_density() {
    run_sparse_density::<f32>();
    if !capability::skip_f64_with_log() {
        run_sparse_density::<f64>();
    }
}

fn run_sparse_density<F>()
where
    F: Float + CubeElement + Pod,
{
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(1, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);

    let density = 0.25_f64;
    let v = ((1.0 / density) / PROP_COMPONENTS as f64).sqrt();

    let mut frac_acc = 0.0_f64;
    for trial in 0..JL_TRIALS {
        let rp = SparseRandomProjection::<F>::builder()
            .n_components(NComponents::Fixed(PROP_COMPONENTS))
            .seed(BASE_SEED + trial as u64)
            .eps(0.1)
            .density(Some(density))
            .build::<F>()
            .expect("sparse build");
        let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
            .expect("sparse fit");
        assert_eq!(rp.density_(), density, "density resolves verbatim");
        let comp: Vec<f64> = rp
            .components(&pool)
            .iter()
            .map(|&val| host_to_f64(val))
            .collect();
        let mut nonzero = 0usize;
        for &e in &comp {
            if e != 0.0 {
                nonzero += 1;
                // Every nonzero entry is exactly ±v.
                assert!(
                    (e.abs() - v).abs() < 1e-4,
                    "Achlioptas nonzero {e} not ±v ({v})"
                );
            }
        }
        frac_acc += nonzero as f64 / comp.len() as f64;
    }
    let frac = frac_acc / JL_TRIALS as f64;
    // Averaged nonzero fraction concentrates at `density` (strict, D-11).
    assert!(
        (frac - density).abs() < 0.01,
        "averaged Achlioptas nonzero fraction {frac} not ≈ density {density}"
    );
}

/// Sparse `transform == X · components_ᵀ` self-consistency (PROPERTY): the device
/// transform equals a host-recomputed `X · components_ᵀ` over the DENSE
/// Achlioptas matrix (same single GEMM as Gaussian; no centering — D-12).
#[test]
fn random_projection_sparse_self_consistency() {
    run_sparse_self_consistency::<f32>();
    if !capability::skip_f64_with_log() {
        run_sparse_self_consistency::<f64>();
    }
}

fn run_sparse_self_consistency<F>()
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_host: Vec<F> = data_matrix::<F>(7, PROP_SAMPLES, PROP_FEATURES);
    let x_dev = DeviceArray::from_host(&mut pool, &x_host);

    // density=None → 1/sqrt(n_features) sklearn default.
    let rp = SparseRandomProjection::<F>::builder()
        .n_components(NComponents::Fixed(PROP_COMPONENTS))
        .seed(BASE_SEED)
        .eps(0.1)
        .density(None)
        .build::<F>()
        .expect("sparse build");
    let rp = TypestateFit::fit(rp, &mut pool, &x_dev, None, (PROP_SAMPLES, PROP_FEATURES))
        .expect("sparse fit");
    let expected_density = 1.0 / (PROP_FEATURES as f64).sqrt();
    assert!(
        (rp.density_() - expected_density).abs() < 1e-9,
        "density=None resolves to 1/sqrt(n_features)"
    );
    let nc = rp.n_components_();

    let z = TypestateTransform::transform(&rp, &mut pool, &x_dev, (PROP_SAMPLES, PROP_FEATURES))
        .expect("transform");
    let z_host: Vec<f64> = z.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
    let comp: Vec<f64> = rp
        .components(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let x64: Vec<f64> = x_host.iter().map(|&v| host_to_f64(v)).collect();

    for r in 0..PROP_SAMPLES {
        for c in 0..nc {
            let mut acc = 0.0_f64;
            for f in 0..PROP_FEATURES {
                acc += x64[r * PROP_FEATURES + f] * comp[c * PROP_FEATURES + f];
            }
            let got = z_host[r * nc + c];
            let tol = 1e-4 + 1e-4 * acc.abs();
            assert!(
                (got - acc).abs() <= tol,
                "sparse self-consistency Z[{r}][{c}] got={got} expected={acc}"
            );
        }
    }
}

/// Sparse JL distortion bound (PROPERTY, averaged, strict — D-11): pairwise
/// distances preserved within the JL bound on average over `JL_TRIALS`.
#[test]
fn random_projection_sparse_jl_distortion() {
    run_jl_distortion::<f32>(true);
    if !capability::skip_f64_with_log() {
        run_jl_distortion::<f64>(true);
    }
}

/// BLDR-01: the zero-arg `new()` and the `builder().build()` default agree on
/// every hyperparameter (the single-source-of-defaults contract, D-08).
#[test]
fn sparse_random_projection_defaults_equal() {
    let from_new = SparseRandomProjection::<f64>::new();
    let from_builder = SparseRandomProjection::<f64>::builder()
        .build::<f64>()
        .expect("default build");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "SparseRandomProjection::new() and builder().build() must share defaults"
    );
}
