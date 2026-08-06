//! NEIGH-PARAMS / NEIGH-RADIUS — `NearestNeighbors`' full hyperparameter surface
//! (`metric`) and its `radius_neighbors` core, against the sklearn oracle.
//!
//! Mirrors `knn_regressor_params_test.rs` (KNN-REG-PARAMS) with the `weights`
//! axis dropped: `NearestNeighbors` has no vote/mean to weight, only a
//! `metric`-parameterized search. Two fixtures:
//!
//! - `nn_params_{f32,f64}_seed42.npz` (`gen_oracle.py::gen_nearest_neighbors_params`)
//!   — `kneighbors` under the five distance FUNCTIONS
//!   (euclidean/manhattan/chebyshev/minkowski/cosine), reusing the
//!   `_knn_reg_data` design (duplicated train pair, coincident queries).
//! - `nn_radius_{f32,f64}_seed42.npz` (`gen_oracle.py::gen_nearest_neighbors_radius`)
//!   — `radius_neighbors` under the same five metrics, stored FLAT + per-row
//!   `radius_counts_<metric>` (a CSR layout without `indptr` — see
//!   `RadiusNeighbors` in `crates/mlrs-algos/src/neighbors/nearest.rs`).
//!
//! Every STRING spelling of `metric`/`algorithm` (the aliasing / tree-name
//! resolution) is Python-shim-only surface — the Rust core never sees a string,
//! only the `Metric` enum — so that coverage lives in
//! `crates/mlrs-py/python/tests/test_oracle_nearest_neighbors_params.py`
//! against the SAME fixtures' `alias_*`/`alg_*` arrays (a second consumer, no
//! regeneration).
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::nearest::NearestNeighbors;
use mlrs_algos::neighbors::Metric;
use mlrs_algos::typestate::{Fit, KNeighbors};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry (`gen_oracle.py::_knn_reg_data`, shared by NEIGH-PARAMS).
const N_TRAIN: usize = 40;
const N_QUERY: usize = 12;
const N_FEATURES: usize = 4;
const K: usize = 5;

/// The fixture's non-degenerate Minkowski exponent (`p != 1, 2, inf`).
const P: f64 = 3.0;

const TOL: f64 = 1e-5;

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
        _ => unreachable!("knn fixtures are f32/f64 only"),
    }
}

fn from_f64<F: Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("knn fixtures are f32/f64 only"),
    }
}

fn fixture_vec<F: Pod>(case: &OracleCase, name: &str) -> Vec<F> {
    case.expect_f64(name)
        .iter()
        .map(|&x| from_f64::<F>(x))
        .collect()
}

fn assert_matches(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length {} != oracle length {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= TOL + TOL * e.abs(),
            "{label}: [{i}] mismatch vs sklearn: got={g:e} expected={e:e} abs_err={abs_err:e}"
        );
    }
}

/// The five `(metric, fixture-key-name)` cases NEIGH-PARAMS covers.
fn metric_cases() -> Vec<(Metric, &'static str)> {
    vec![
        (Metric::Euclidean, "euclidean"),
        (Metric::Manhattan, "manhattan"),
        (Metric::Chebyshev, "chebyshev"),
        (Metric::Minkowski { p: P }, "minkowski"),
        (Metric::Cosine, "cosine"),
    ]
}

/// Does the Minkowski case have to be skipped on this backend? Mirrors
/// `knn_regressor_params_test.rs::skip_minkowski`.
fn skip_minkowski<F>(metric: Metric) -> bool
where
    F: Float + CubeElement + Pod,
{
    matches!(metric, Metric::Minkowski { .. })
        && capability::guard_f64_transcendental::<F>("nearest_neighbors_params_test").is_err()
}

// ---------------------------------------------------------------------------
// Fixture integrity
// ---------------------------------------------------------------------------

#[test]
fn fixture_loads() {
    let case = load_npz(fixture("nn_params_f64_seed42.npz")).expect("load fixture");
    assert_eq!(case.expect_f64("X").len(), N_TRAIN * N_FEATURES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("k")[0] as usize, K);
    assert_eq!(case.expect_f64("p")[0], P);
    for (_, name) in metric_cases() {
        assert_eq!(
            case.expect_f64(&format!("distances_{name}")).len(),
            N_QUERY * K,
            "distances_{name}"
        );
        assert_eq!(
            case.expect_f64(&format!("indices_{name}")).len(),
            N_QUERY * K,
            "indices_{name}"
        );
    }
}

#[test]
fn radius_fixture_loads() {
    let case = load_npz(fixture("nn_radius_f64_seed42.npz")).expect("load radius fixture");
    assert_eq!(case.expect_f64("X").len(), N_TRAIN * N_FEATURES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    for (_, name) in metric_cases() {
        let counts = case.expect_f64(&format!("radius_counts_{name}"));
        assert_eq!(counts.len(), N_QUERY, "radius_counts_{name}");
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        assert_eq!(
            case.expect_f64(&format!("radius_distances_{name}")).len(),
            total,
            "radius_distances_{name}"
        );
        assert_eq!(
            case.expect_f64(&format!("radius_indices_{name}")).len(),
            total,
            "radius_indices_{name}"
        );
        // A degenerate radius (0 or every-point) would make this test vacuous
        // for that metric — at least one row must have a PARTIAL match set.
        assert!(
            counts.iter().any(|&c| c > 0.0 && (c as usize) < N_TRAIN),
            "radius_counts_{name} must contain at least one partial (non-empty, \
             non-total) row, or the radius threshold is not being exercised"
        );
    }
}

// ---------------------------------------------------------------------------
// Builder validation (data-INDEPENDENT, no device)
// ---------------------------------------------------------------------------

/// BLDR-01, extended to `metric` (NEIGH-PARAMS): `new()` and
/// `builder().build()?` must agree on the WHOLE hyperparameter set.
#[test]
fn defaults_equal_including_metric() {
    let from_new = NearestNeighbors::<f64>::new();
    let from_builder = NearestNeighbors::<f64>::builder()
        .build::<f64>()
        .expect("default NearestNeighborsBuilder builds");
    assert!(from_new.hyperparams_eq(&from_builder));
}

/// The builder round-trips `metric`, so a value set on it is the value the
/// estimator computes with.
#[test]
fn builder_round_trips_metric() {
    let est = NearestNeighbors::<f64>::builder()
        .n_neighbors(7)
        .metric(Metric::Minkowski { p: 2.5 })
        .build::<f64>()
        .expect("valid hyperparameters build");
    assert_eq!(est.n_neighbors(), 7);
    // `metric` is private to the Fitted/Unfit structs; round-trip is asserted
    // behaviourally via `kneighbors` in `metric_matrix_match_sklearn_*` below,
    // which would fail if the builder dropped the value.
}

// ---------------------------------------------------------------------------
// kneighbors: metric matrix against sklearn
// ---------------------------------------------------------------------------

fn check_kneighbors_case<F>(fixture_name: &str, metric: Metric, oracle_key: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load nn_params fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let expected_dist: Vec<f64> = case
        .expect_f64(&format!("distances_{oracle_key}"))
        .to_vec();
    let expected_idx: Vec<f64> = case
        .expect_f64(&format!("indices_{oracle_key}"))
        .to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<F>::builder()
        .n_neighbors(K)
        .metric(metric)
        .build::<F>()
        .expect("build NearestNeighbors")
        .fit(&mut pool, &x_dev, None, (N_TRAIN, N_FEATURES))
        .expect("fit on valid geometry");

    let (dist_dev, idx_dev) = nn
        .kneighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), K)
        .expect("kneighbors on valid geometry");
    let got_dist: Vec<f64> = dist_dev
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    let got_idx: Vec<i32> = idx_dev.to_host(&pool);
    dist_dev.release_into(&mut pool);
    idx_dev.release_into(&mut pool);

    assert_matches(&got_dist, &expected_dist, &format!("distances_{oracle_key}"));

    // Indices compared as a per-row SET (see `knn_regressor_params_test.rs`'s
    // identical rationale): the fixture's duplicated training pair is an exact
    // distance tie whose relative order is not determined by the problem.
    for q in 0..N_QUERY {
        let mut got: Vec<i64> = got_idx[q * K..(q + 1) * K].iter().map(|&v| v as i64).collect();
        let mut want: Vec<i64> = expected_idx[q * K..(q + 1) * K]
            .iter()
            .map(|&v| v.round() as i64)
            .collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "indices_{oracle_key} row {q}");
    }
}

fn run_all_metric_cases<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    for (metric, name) in metric_cases() {
        if skip_minkowski::<F>(metric) {
            println!("kneighbors metric={name}: SKIPPED (no f64 transcendentals on this backend)");
            continue;
        }
        check_kneighbors_case::<F>(fixture_name, metric, name);
    }
}

#[test]
fn metric_matrix_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "nn_params");
    run_all_metric_cases::<f32>("nn_params_f32_seed42.npz");
}

#[test]
fn metric_matrix_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "nn_params");
    if capability::skip_f64_with_log() {
        println!("nearest_neighbors params f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_all_metric_cases::<f64>("nn_params_f64_seed42.npz");
}

// ---------------------------------------------------------------------------
// radius_neighbors: metric matrix against sklearn (NEIGH-RADIUS)
// ---------------------------------------------------------------------------

/// `radius_neighbors` under `metric` matches the sklearn oracle EXACTLY: same
/// per-row match COUNT, same indices in the SAME ascending order (both sides
/// scan train points column-by-column and keep the ones within `radius` — see
/// `RadiusNeighbors`' module docs — so, unlike `kneighbors`' top-k tie-break,
/// there is no ordering ambiguity to tolerate), and distances within 1e-5.
fn check_radius_case<F>(fixture_name: &str, metric: Metric, name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load nn_radius fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let radius = case.expect_f64(&format!("radius_{name}"))[0];
    let expected_counts: Vec<f64> = case.expect_f64(&format!("radius_counts_{name}")).to_vec();
    let expected_dist: Vec<f64> = case.expect_f64(&format!("radius_distances_{name}")).to_vec();
    let expected_idx: Vec<f64> = case.expect_f64(&format!("radius_indices_{name}")).to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<F>::builder()
        .metric(metric)
        .build::<F>()
        .expect("build NearestNeighbors")
        .fit(&mut pool, &x_dev, None, (N_TRAIN, N_FEATURES))
        .expect("fit on valid geometry");

    let result = nn
        .radius_neighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), radius)
        .expect("radius_neighbors on valid geometry");
    xq_dev.release_into(&mut pool);

    assert_eq!(
        result.counts.len(),
        N_QUERY,
        "radius_neighbors must return one count per query row"
    );
    let mut offset = 0usize;
    for q in 0..N_QUERY {
        let want_count = expected_counts[q] as usize;
        assert_eq!(
            result.counts[q] as usize, want_count,
            "radius_counts_{name} row {q}: match count mismatch"
        );
        let got_dist: Vec<f64> = result.distances[offset..offset + want_count]
            .iter()
            .map(|&v| host_to_f64(v))
            .collect();
        let got_idx: Vec<i32> = result.indices[offset..offset + want_count].to_vec();
        let want_dist = &expected_dist[offset..offset + want_count];
        let want_idx: Vec<i32> = expected_idx[offset..offset + want_count]
            .iter()
            .map(|&v| v.round() as i32)
            .collect();

        assert_eq!(got_idx, want_idx, "radius_indices_{name} row {q}: order/set mismatch");
        assert_matches(&got_dist, want_dist, &format!("radius_distances_{name} row {q}"));

        offset += want_count;
    }
}

fn run_all_radius_cases<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    for (metric, name) in metric_cases() {
        if skip_minkowski::<F>(metric) {
            println!("radius_neighbors metric={name}: SKIPPED (no f64 transcendentals on this backend)");
            continue;
        }
        check_radius_case::<F>(fixture_name, metric, name);
    }
}

#[test]
fn radius_neighbors_metric_matrix_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "nn_radius");
    run_all_radius_cases::<f32>("nn_radius_f32_seed42.npz");
}

#[test]
fn radius_neighbors_metric_matrix_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "nn_radius");
    if capability::skip_f64_with_log() {
        println!("nearest_neighbors radius f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_all_radius_cases::<f64>("nn_radius_f64_seed42.npz");
}

// ---------------------------------------------------------------------------
// radius_neighbors: validation + edge cases
// ---------------------------------------------------------------------------

/// ASVS V5: a negative `radius` is rejected BEFORE any prim launch, as
/// `AlgoError::InvalidEps` (radius shares DBSCAN's "non-negative distance
/// threshold" contract).
#[test]
fn radius_neighbors_rejects_negative_radius() {
    use mlrs_algos::error::AlgoError;

    let case = load_npz(fixture("nn_radius_f32_seed42.npz")).expect("load fixture");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let xq: Vec<f32> = fixture_vec::<f32>(&case, "Xq");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<f32>::builder()
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (N_TRAIN, N_FEATURES))
        .expect("fit");

    match nn.radius_neighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), -1.0) {
        Err(AlgoError::InvalidEps { eps, .. }) => assert_eq!(eps, -1.0),
        Err(other) => panic!("negative radius must be AlgoError::InvalidEps, got {other:?}"),
        Ok(_) => panic!("negative radius must be rejected before launch, got Ok"),
    }
}

/// `radius = 0.0` is a legal query: an empty match set (unless a training point
/// coincides exactly with a query point) is a valid answer, not an error.
#[test]
fn radius_neighbors_accepts_zero_radius() {
    let case = load_npz(fixture("nn_radius_f32_seed42.npz")).expect("load fixture");
    let x: Vec<f32> = fixture_vec::<f32>(&case, "X");
    let xq: Vec<f32> = fixture_vec::<f32>(&case, "Xq");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &xq);

    let nn = NearestNeighbors::<f32>::builder()
        .build::<f32>()
        .expect("build")
        .fit(&mut pool, &x_dev, None, (N_TRAIN, N_FEATURES))
        .expect("fit");

    let result = nn
        .radius_neighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), 0.0)
        .expect("radius=0.0 must be accepted");
    assert_eq!(result.counts.len(), N_QUERY);
}
