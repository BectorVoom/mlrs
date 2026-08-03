//! KNN-REG-PARAMS — `KNeighborsRegressor`'s full hyperparameter surface against
//! the sklearn oracle.
//!
//! Every case loads the committed `knn_reg_params_{f32,f64}_seed42.npz` fixture
//! (`scripts/gen_oracle.py::gen_knn_regressor_params`) and asserts `predict`
//! matches `sklearn.neighbors.KNeighborsRegressor(algorithm='brute', ...)`
//! within 1e-5 for:
//!
//! - `weights` ∈ {uniform, distance} × `metric` ∈ {euclidean, manhattan,
//!   chebyshev, minkowski(p=3), cosine} — all ten combinations;
//! - a MULTI-OUTPUT (`n_train × 2`) target under both weightings;
//! - `kneighbors` under a non-default metric (the one place a metric bug can
//!   hide behind a `predict` that happens to average to the right number).
//!
//! The fixture's query set deliberately contains rows that COINCIDE with
//! training points, and one duplicated training row, so every
//! `weights='distance'` case exercises the `1/0` indicator branch rather than
//! only the generic `1/d` path. That branch is the one whose failure mode is
//! silent (`inf/inf` → NaN), so it must not be reachable only by accident.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
use mlrs_algos::neighbors::{Metric, Weights};
use mlrs_algos::typestate::{Fit, KNeighbors, Predict};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry (`gen_oracle.py::_knn_reg_data`).
const N_TRAIN: usize = 40;
const N_QUERY: usize = 12;
const N_FEATURES: usize = 4;
const K: usize = 5;
const N_OUTPUTS: usize = 2;

/// The fixture's non-degenerate Minkowski exponent (`p != 1, 2, inf`, so it
/// reaches the general `minkowski_dist` kernel rather than a collapsed fast
/// path).
const P: f64 = 3.0;

const PRED_TOL: f64 = 1e-5;

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

/// Compare a flat prediction buffer against the fixture's sklearn reference,
/// with the project's mixed abs/rel 1e-5 bar.
fn assert_matches(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: prediction length {} != oracle length {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= PRED_TOL + PRED_TOL * e.abs(),
            "{label}: predict[{i}] mismatch vs sklearn: got={g:e} expected={e:e} \
             abs_err={abs_err:e}"
        );
    }
}

/// The fixture's `y_multi` is stored `n_train × n_outputs` row-major, which is
/// exactly the layout `fit` infers `n_outputs` from — no transposition needed.
/// Named so a future column-major fixture change fails here rather than
/// silently predicting from a shuffled target.
fn multi_target<F: Pod>(case: &OracleCase) -> Vec<F> {
    let v = fixture_vec::<F>(case, "y_multi");
    assert_eq!(
        v.len(),
        N_TRAIN * N_OUTPUTS,
        "y_multi must be n_train x n_outputs row-major"
    );
    v
}

/// Shared body: fit under `(weights, metric)` on the given target, predict the
/// fixture's query set, compare against `oracle_key`.
fn check_case<F>(
    fixture_name: &str,
    weights: Weights,
    metric: Metric,
    target_key: &str,
    oracle_key: &str,
) where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn_reg_params fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let y: Vec<F> = if target_key == "y_multi" {
        multi_target::<F>(&case)
    } else {
        fixture_vec::<F>(&case, target_key)
    };
    let expected: Vec<f64> = case.expect_f64(oracle_key).to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let reg = KNeighborsRegressor::<F>::builder()
        .n_neighbors(K)
        .weights(weights)
        .metric(metric)
        .build::<F>()
        .expect("build KNeighborsRegressor")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
        .expect("fit on valid geometry");

    let pred_dev = reg
        .predict(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict on valid geometry");
    let got: Vec<f64> = pred_dev
        .to_host(&pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    pred_dev.release_into(&mut pool);

    assert_matches(&got, &expected, oracle_key);
}

/// The ten `(metric, weights)` combinations, paired with the fixture key
/// holding sklearn's answer for each.
fn metric_cases() -> Vec<(Metric, Weights, String)> {
    let mut out = Vec::new();
    for (metric, name) in [
        (Metric::Euclidean, "euclidean"),
        (Metric::Manhattan, "manhattan"),
        (Metric::Chebyshev, "chebyshev"),
        (Metric::Minkowski { p: P }, "minkowski"),
        (Metric::Cosine, "cosine"),
    ] {
        for (weights, wname) in [
            (Weights::Uniform, "uniform"),
            (Weights::Distance, "distance"),
        ] {
            out.push((metric, weights, format!("predict_{name}_{wname}")));
        }
    }
    out
}

/// Does the Minkowski case have to be skipped on this backend?
///
/// `minkowski_dist` is the only metric kernel that evaluates `F::powf`, and on
/// a backend with f64 arithmetic but no f64 transcendentals that SEGFAULTS the
/// driver's shader compiler rather than failing the launch — so the prim
/// capability-gates it. The gate returns a typed error, which would fail this
/// test as a mismatch; skipping keeps the signal honest (the other four metrics
/// still run at f64 there).
fn skip_minkowski<F>(metric: Metric) -> bool
where
    F: Float + CubeElement + Pod,
{
    matches!(metric, Metric::Minkowski { .. })
        && capability::guard_f64_transcendental::<F>("knn_regressor_params_test").is_err()
}

fn run_all_metric_cases<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    for (metric, weights, key) in metric_cases() {
        if skip_minkowski::<F>(metric) {
            println!("{key}: SKIPPED (no f64 transcendentals on this backend)");
            continue;
        }
        check_case::<F>(fixture_name, weights, metric, "y", &key);
    }
}

// ---------------------------------------------------------------------------
// Fixture integrity
// ---------------------------------------------------------------------------

/// LOAD-NOT-JUST-PRESENT: every array the parameter cases read is present and
/// the right length, and the fixture really does contain the coincident query
/// rows the `weights='distance'` branch needs. Without that last assert the ten
/// distance cases could all pass while never once dividing by zero — the exact
/// situation the branch exists for.
#[test]
fn fixture_loads() {
    let case = load_npz(fixture("knn_reg_params_f64_seed42.npz")).expect("load fixture");
    assert_eq!(case.expect_f64("X").len(), N_TRAIN * N_FEATURES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_TRAIN);
    assert_eq!(case.expect_f64("y_multi").len(), N_TRAIN * N_OUTPUTS);
    assert_eq!(case.expect_f64("k")[0] as usize, K);
    assert_eq!(case.expect_f64("p")[0], P);
    for (_, _, key) in metric_cases() {
        assert_eq!(case.expect_f64(&key).len(), N_QUERY, "{key}");
    }
    for key in ["predict_multi_uniform", "predict_multi_distance"] {
        assert_eq!(case.expect_f64(key).len(), N_QUERY * N_OUTPUTS, "{key}");
    }
    assert_eq!(case.expect_f64("distances_manhattan").len(), N_QUERY * K);
    assert_eq!(case.expect_f64("indices_manhattan").len(), N_QUERY * K);

    // A query row that EXACTLY equals a training row — the `1/0` weighting
    // branch's precondition.
    let x = case.expect_f64("X");
    let xq = case.expect_f64("Xq");
    let coincident = (0..N_QUERY).filter(|&q| {
        (0..N_TRAIN).any(|t| {
            (0..N_FEATURES).all(|f| x[t * N_FEATURES + f] == xq[q * N_FEATURES + f])
        })
    });
    assert!(
        coincident.count() >= 2,
        "the fixture must contain at least two queries coincident with a training \
         point, or every weights='distance' case silently skips the 1/0 branch"
    );
}

// ---------------------------------------------------------------------------
// Builder validation (data-INDEPENDENT, no device)
// ---------------------------------------------------------------------------

/// BLDR-01: `new()` equals `builder().build()?` across the WHOLE hyperparameter
/// set, not just `n_neighbors`.
#[test]
fn defaults_equal() {
    let from_new = KNeighborsRegressor::<f64>::new();
    let from_builder = KNeighborsRegressor::<f64>::builder()
        .build::<f64>()
        .expect("default KNeighborsRegressorBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "new() and builder().build()? must agree on every hyperparameter (BLDR-01)"
    );
    assert_eq!(from_new.weights(), Weights::Uniform);
    assert_eq!(from_new.metric(), Metric::Euclidean);
    assert_eq!(from_new.n_neighbors(), 5);
}

/// The builder round-trips every hyperparameter, so a value set on it is the
/// value the estimator computes with (a setter that dropped its argument would
/// otherwise only show up as a numeric mismatch in some oracle case).
#[test]
fn builder_round_trips_every_param() {
    let est = KNeighborsRegressor::<f64>::builder()
        .n_neighbors(11)
        .weights(Weights::Distance)
        .metric(Metric::Minkowski { p: 2.5 })
        .build::<f64>()
        .expect("valid hyperparameters build");
    assert_eq!(est.n_neighbors(), 11);
    assert_eq!(est.weights(), Weights::Distance);
    assert_eq!(est.metric(), Metric::Minkowski { p: 2.5 });
}

/// D-08 / ASVS V5: a Minkowski `p < 1` is rejected at BUILD, before any data is
/// seen. `p < 1` is not a metric (the triangle inequality fails) and the kernel
/// would return a finite but meaningless ordering rather than failing.
#[test]
fn builder_rejects_minkowski_p_below_one() {
    let err = KNeighborsRegressor::<f64>::builder()
        .metric(Metric::Minkowski { p: 0.5 })
        .build::<f64>();
    assert!(err.is_err(), "p = 0.5 must be rejected at build");
}

/// A NaN exponent must be rejected too. This is the case the natural `p < 1.0`
/// spelling gets wrong: every ordered comparison against NaN is false, so `p <
/// 1.0` waves it through and the kernel emits NaN distances that `top_k` orders
/// arbitrarily — a silently wrong neighbour set, not an error.
#[test]
fn builder_rejects_minkowski_p_nan() {
    let err = KNeighborsRegressor::<f64>::builder()
        .metric(Metric::Minkowski { p: f64::NAN })
        .build::<f64>();
    assert!(err.is_err(), "p = NaN must be rejected at build");
}

/// `n_neighbors = 0` stays rejected at build (unchanged by KNN-REG-PARAMS).
#[test]
fn builder_rejects_zero_n_neighbors() {
    assert!(KNeighborsRegressor::<f64>::builder()
        .n_neighbors(0)
        .build::<f64>()
        .is_err());
}

// ---------------------------------------------------------------------------
// weights x metric against sklearn
// ---------------------------------------------------------------------------

#[test]
fn weights_metric_matrix_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "params");
    run_all_metric_cases::<f32>("knn_reg_params_f32_seed42.npz");
}

#[test]
fn weights_metric_matrix_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "params");
    if capability::skip_f64_with_log() {
        println!("knn_regressor params f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_all_metric_cases::<f64>("knn_reg_params_f64_seed42.npz");
}

// ---------------------------------------------------------------------------
// Multi-output targets
// ---------------------------------------------------------------------------

#[test]
fn multi_output_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    for (weights, key) in [
        (Weights::Uniform, "predict_multi_uniform"),
        (Weights::Distance, "predict_multi_distance"),
    ] {
        check_case::<f32>(
            "knn_reg_params_f32_seed42.npz",
            weights,
            Metric::Euclidean,
            "y_multi",
            key,
        );
    }
}

#[test]
fn multi_output_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        println!("knn_regressor multi-output f64: SKIPPED (no f64 support)");
        return;
    }
    for (weights, key) in [
        (Weights::Uniform, "predict_multi_uniform"),
        (Weights::Distance, "predict_multi_distance"),
    ] {
        check_case::<f64>(
            "knn_reg_params_f64_seed42.npz",
            weights,
            Metric::Euclidean,
            "y_multi",
            key,
        );
    }
}

/// `n_outputs_` is INFERRED from `y.len() / n_train`, so a 1-D target and a
/// 2-column one must land on 1 and 2 without the caller saying so.
#[test]
fn n_outputs_is_inferred_from_y() {
    let case = load_npz(fixture("knn_reg_params_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    for (key, expect) in [("y", 1usize), ("y_multi", 2usize)] {
        let y: Vec<f64> = fixture_vec::<f64>(&case, key);
        let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
        let reg = KNeighborsRegressor::<f64>::builder()
            .build::<f64>()
            .expect("build")
            .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
            .expect("fit");
        assert_eq!(reg.n_outputs(), expect, "n_outputs for '{key}'");
        assert_eq!(reg.train_shape(), (N_TRAIN, N_FEATURES));
    }
}

/// ASVS V5: a `y` whose length is not a whole multiple of `n_train` is a typed
/// error, not a fractional inferred width that would mis-stride every gather.
#[test]
fn fit_rejects_ragged_y() {
    let case = load_npz(fixture("knn_reg_params_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    // N_TRAIN + 1 elements: neither 1 nor 2 outputs' worth.
    let ragged = vec![1.0f64; N_TRAIN + 1];
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &ragged);
    let err = KNeighborsRegressor::<f64>::builder()
        .build::<f64>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES));
    assert!(err.is_err(), "a y of length n_train + 1 must be rejected");
}

// ---------------------------------------------------------------------------
// fit_owned (KNN-REG-FIT): ownership transfer instead of a device copy
// ---------------------------------------------------------------------------

/// `fit_owned` and the borrowing `Fit::fit` must produce the SAME predictions.
///
/// They differ only in how the operands got onto the device — one copies, the
/// other adopts — so any divergence means the ownership path stored something
/// different (a wrong stride, a wrong `n_outputs`, an aliased buffer).
#[test]
fn fit_owned_matches_borrowing_fit() {
    let case = load_npz(fixture("knn_reg_params_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let xq: Vec<f64> = fixture_vec::<f64>(&case, "Xq");
    let y: Vec<f64> = multi_target::<f64>(&case);
    let expected: Vec<f64> = case.expect_f64("predict_multi_distance").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let xq_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xq);

    let build = || {
        KNeighborsRegressor::<f64>::builder()
            .n_neighbors(K)
            .weights(Weights::Distance)
            .metric(Metric::Euclidean)
            .build::<f64>()
            .expect("build")
    };

    // Borrowing path: the estimator copies, and the caller RELEASES its inputs
    // afterwards — the exact sequence that would corrupt a handle-aliasing
    // implementation, because the next `acquire` of that size hands the freed
    // buffer straight back.
    let borrowed = {
        let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
        let est = build()
            .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
            .expect("borrowing fit");
        x_dev.release_into(&mut pool);
        y_dev.release_into(&mut pool);
        est
    };

    // Owning path: the caller gives the arrays up entirely.
    let owned = {
        let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
        let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);
        build()
            .fit_owned(&mut pool, x_dev, y_dev, (N_TRAIN, N_FEATURES))
            .expect("owning fit")
    };

    assert_eq!(borrowed.n_outputs(), owned.n_outputs());
    assert_eq!(borrowed.train_shape(), owned.train_shape());

    for (label, est) in [("borrowed", &borrowed), ("owned", &owned)] {
        let pred = est
            .predict(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
            .expect("predict");
        let got: Vec<f64> = pred.to_host(&pool).iter().map(|&v| host_to_f64(v)).collect();
        pred.release_into(&mut pool);
        assert_matches(&got, &expected, label);
    }
}

/// A rejected `fit_owned` must RELEASE the operands it was handed.
///
/// The caller has already given them up, so nothing else can return them to the
/// pool — leaking them would raise `live_bytes` permanently, which is the
/// FOUND-05 conservation property the D-10 memory gate asserts on.
///
/// The operands are built with `pool.acquire` + `from_raw` rather than
/// `from_host`, because `from_host` acquires a metering handle and releases it
/// again immediately (CubeCL 0.10 has no in-place write into an `empty`
/// handle), so an uploaded array contributes NOTHING to `live_bytes` and this
/// test could not observe a leak through one. Pool-acquired buffers are the
/// form the counter actually tracks — and the form every device-resident
/// producer inside the prims layer hands around. Their contents are irrelevant:
/// the fit is rejected on geometry, before either is read.
#[test]
fn fit_owned_releases_operands_on_rejection() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let before = pool.stats().live_bytes;

    let x_len = N_TRAIN * N_FEATURES;
    // Length N_TRAIN + 1: not a whole multiple of n_train, so `n_outputs`
    // cannot be inferred and the fit must be rejected.
    let y_len = N_TRAIN + 1;
    let x_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_raw(pool.acquire(x_len * size_of::<f64>()), x_len);
    let y_dev: DeviceArray<ActiveRuntime, f64> =
        DeviceArray::from_raw(pool.acquire(y_len * size_of::<f64>()), y_len);
    assert!(
        pool.stats().live_bytes > before,
        "the two acquisitions must be metered, or this test cannot observe a leak"
    );

    let err = KNeighborsRegressor::<f64>::builder()
        .build::<f64>()
        .expect("build")
        .fit_owned(&mut pool, x_dev, y_dev, (N_TRAIN, N_FEATURES));
    assert!(err.is_err(), "a ragged y must be rejected");
    assert_eq!(
        pool.stats().live_bytes,
        before,
        "fit_owned must return both operands to the pool when it rejects them"
    );
}

// ---------------------------------------------------------------------------
// kneighbors under a non-default metric
// ---------------------------------------------------------------------------

/// `kneighbors` must report the CONFIGURED metric's neighbours, not Euclidean
/// ones. Checked separately from `predict` because a wrong-metric neighbour set
/// can still average to a nearly-right prediction on smooth targets — this is
/// the assert that actually pins the metric.
#[test]
fn kneighbors_uses_configured_metric_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        println!("knn_regressor kneighbors f64: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("knn_reg_params_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let xq: Vec<f64> = fixture_vec::<f64>(&case, "Xq");
    let y: Vec<f64> = fixture_vec::<f64>(&case, "y");
    let ref_dist = case.expect_f64("distances_manhattan").to_vec();
    let ref_idx = case.expect_f64("indices_manhattan").to_vec();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let reg = KNeighborsRegressor::<f64>::builder()
        .n_neighbors(K)
        .metric(Metric::Manhattan)
        .build::<f64>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
        .expect("fit");

    let (dist, idx) = reg
        .kneighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), K)
        .expect("kneighbors on valid geometry");
    let got_dist: Vec<f64> = dist.to_host(&pool);
    let got_idx: Vec<i32> = idx.to_host(&pool);
    dist.release_into(&mut pool);
    idx.release_into(&mut pool);

    assert_matches(&got_dist, &ref_dist, "distances_manhattan");

    // Indices are compared as a per-row SET, not position by position. The
    // fixture contains a duplicated training row (deliberately — it is what
    // gives a coincident query two zero-distance neighbours), so those two
    // indices are an exact distance TIE and their relative order is not
    // determined by the problem. mlrs's `top_k` resolves ties to the lowest
    // index; sklearn's brute path makes no such guarantee and here returns the
    // other order. Asserting positional equality would be asserting an
    // implementation detail of sklearn's sort. The distances, which ARE fully
    // determined, are compared elementwise above.
    for q in 0..N_QUERY {
        let mut got: Vec<i64> = got_idx[q * K..(q + 1) * K].iter().map(|&v| v as i64).collect();
        let mut want: Vec<i64> = ref_idx[q * K..(q + 1) * K].iter().map(|&v| v as i64).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "indices_manhattan row {q}");
    }
}
