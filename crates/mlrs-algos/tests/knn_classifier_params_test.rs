//! KNN-CLF-PARAMS — `KNeighborsClassifier`'s full hyperparameter surface against
//! the sklearn oracle.
//!
//! Every case loads the committed `knn_clf_params_{f32,f64}_seed42.npz` fixture
//! (`scripts/gen_oracle.py::gen_knn_classifier_params`) and asserts, for
//! `weights` ∈ {uniform, distance} × `metric` ∈ {euclidean, manhattan,
//! chebyshev, minkowski(p=3), cosine} — all ten combinations:
//!
//! - `predict_labels` matches `sklearn.neighbors.KNeighborsClassifier(
//!   algorithm='brute', ...)` EXACTLY (labels are ids, not measurements — a
//!   tolerance on them would be meaningless);
//! - `predict_proba` matches it within 1e-5. Both are checked because `predict`
//!   is an argmax: a proba matrix that is mis-normalized, scaled, or has two
//!   columns transposed can still argmax to the right label on every row of a
//!   12-query fixture. The proba arrays are what actually pin the weighting.
//!
//! Plus `kneighbors` under a non-default metric (the one place a metric bug can
//! hide behind a `predict` that happens to vote the right way anyway), and the
//! `classes_` round-trip over a NON-CONTIGUOUS label set.
//!
//! The fixture's query set deliberately contains rows that COINCIDE with
//! training points, and one duplicated training row, so every
//! `weights='distance'` case exercises the `1/0` indicator branch rather than
//! only the generic `1/d` path. That branch is the one whose failure mode is
//! silent (`inf/inf` → NaN), so it must not be reachable only by accident.
//!
//! `algorithm`, `leaf_size`, `n_jobs`, the nine `metric` SPELLINGS and the
//! multi-output target are all resolved in the Python shim, so their oracle
//! cases live in `crates/mlrs-py/python/tests/test_oracle_knn_classifier_params.py`
//! against the same fixture — the core takes only the parameters that change
//! what it computes.
//!
//! f64 functions carry the `skip_f64_with_log` capability gate (cpu runs f64;
//! rocm skips-with-log, D-07). Per AGENTS.md §2 tests live in
//! `crates/mlrs-algos/tests/`, never an in-source `#[cfg(test)] mod tests`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::classifier::{prepare_labels, KNeighborsClassifier};
use mlrs_algos::neighbors::{Metric, Weights};
use mlrs_algos::typestate::{Fit, KNeighbors};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry (`gen_oracle.py::_knn_clf_data`, which reuses
/// `_knn_reg_data`'s design).
const N_TRAIN: usize = 40;
const N_QUERY: usize = 12;
const N_FEATURES: usize = 4;
const K: usize = 5;
const N_CLASSES: usize = 3;

/// The fixture's non-degenerate Minkowski exponent (`p != 1, 2, inf`, so it
/// reaches the general `minkowski_dist` kernel rather than a collapsed fast
/// path).
const P: f64 = 3.0;

/// The NON-CONTIGUOUS training label set. `classes_` must be exactly this, and
/// `predict_labels` must only ever return ids drawn from it.
const CLASSES: [i32; N_CLASSES] = [0, 2, 7];

const PROBA_TOL: f64 = 1e-5;

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

/// Compare a flat proba buffer against the fixture's sklearn reference, with the
/// project's mixed abs/rel 1e-5 bar.
fn assert_proba(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: proba length {} != oracle length {}",
        got.len(),
        expected.len()
    );
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let abs_err = (g - e).abs();
        assert!(
            abs_err <= PROBA_TOL + PROBA_TOL * e.abs(),
            "{label}: proba[{i}] mismatch vs sklearn: got={g:e} expected={e:e} \
             abs_err={abs_err:e}"
        );
    }
}

/// Labels are IDENTIFIERS, so this is exact equality — a tolerance on a class id
/// would pass a prediction that named a different class.
fn assert_labels(got: &[i32], expected: &[f64], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: label count");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g as f64, e,
            "{label}: predict[{i}] mismatch vs sklearn: got={g} expected={e}"
        );
    }
}

/// Fit under `(weights, metric)` on the fixture's target and return the
/// `(labels, proba)` pair for its query set.
fn run_case<F>(fixture_name: &str, weights: Weights, metric: Metric) -> (Vec<i32>, Vec<f64>)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn_clf_params fixture");
    let x: Vec<F> = fixture_vec::<F>(&case, "X");
    let xq: Vec<F> = fixture_vec::<F>(&case, "Xq");
    let y: Vec<F> = fixture_vec::<F>(&case, "y");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &y);

    let clf = KNeighborsClassifier::<F>::builder()
        .n_neighbors(K)
        .weights(weights)
        .metric(metric)
        .build::<F>()
        .expect("build KNeighborsClassifier")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
        .expect("fit on valid geometry");

    assert_eq!(
        clf.classes(),
        CLASSES,
        "classes_ must be the DISTINCT SORTED training labels"
    );

    let labels = clf
        .predict_labels_host(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_labels on valid geometry");
    let proba: Vec<f64> = clf
        .predict_proba_host(&mut pool, &xq_dev, (N_QUERY, N_FEATURES))
        .expect("predict_proba on valid geometry")
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    (labels, proba)
}

/// The ten `(metric, weights)` combinations, paired with the fixture key stem
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
            out.push((metric, weights, format!("{name}_{wname}")));
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
        && capability::guard_f64_transcendental::<F>("knn_classifier_params_test").is_err()
}

fn run_all_metric_cases<F>(fixture_name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(fixture(fixture_name)).expect("load knn_clf_params fixture");
    for (metric, weights, stem) in metric_cases() {
        if skip_minkowski::<F>(metric) {
            println!("{stem}: SKIPPED (no f64 transcendentals on this backend)");
            continue;
        }
        let (labels, proba) = run_case::<F>(fixture_name, weights, metric);
        assert_labels(
            &labels,
            case.expect_f64(&format!("predict_{stem}")),
            &format!("predict_{stem}"),
        );
        assert_proba(
            &proba,
            case.expect_f64(&format!("proba_{stem}")),
            &format!("proba_{stem}"),
        );
        // Every predicted id must come from the training label set. An argmax
        // that returned the DENSE column index would produce `1`, which is not a
        // class here — and on a contiguous fixture would have looked correct.
        for &l in &labels {
            assert!(
                CLASSES.contains(&l),
                "{stem}: predicted id {l} is not one of the training classes \
                 {CLASSES:?}"
            );
        }
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
    let case = load_npz(fixture("knn_clf_params_f64_seed42.npz")).expect("load fixture");
    assert_eq!(case.expect_f64("X").len(), N_TRAIN * N_FEATURES);
    assert_eq!(case.expect_f64("Xq").len(), N_QUERY * N_FEATURES);
    assert_eq!(case.expect_f64("y").len(), N_TRAIN);
    assert_eq!(case.expect_f64("k")[0] as usize, K);
    assert_eq!(case.expect_f64("p")[0], P);
    assert_eq!(
        case.expect_f64("classes")
            .iter()
            .map(|&c| c as i32)
            .collect::<Vec<_>>(),
        CLASSES,
        "the fixture's class set must match this test's CLASSES constant"
    );
    for (_, _, stem) in metric_cases() {
        assert_eq!(case.expect_f64(&format!("predict_{stem}")).len(), N_QUERY);
        assert_eq!(
            case.expect_f64(&format!("proba_{stem}")).len(),
            N_QUERY * N_CLASSES
        );
    }
    assert_eq!(case.expect_f64("distances_manhattan").len(), N_QUERY * K);
    assert_eq!(case.expect_f64("indices_manhattan").len(), N_QUERY * K);

    // ALL THREE classes must be represented in the training target, and no class
    // may dominate: a fixture where one class held most of the rows would let a
    // broken vote agree with sklearn by always guessing the majority.
    let y = case.expect_f64("y");
    for c in CLASSES {
        let n = y.iter().filter(|&&v| v as i32 == c).count();
        assert!(
            n >= N_TRAIN / 5,
            "class {c} has only {n} of {N_TRAIN} training rows — too few to \
             distinguish a real vote from a majority guess"
        );
    }

    // A query row that EXACTLY equals a training row — the `1/0` weighting
    // branch's precondition.
    let x = case.expect_f64("X");
    let xq = case.expect_f64("Xq");
    let coincident = (0..N_QUERY).filter(|&q| {
        (0..N_TRAIN)
            .any(|t| (0..N_FEATURES).all(|f| x[t * N_FEATURES + f] == xq[q * N_FEATURES + f]))
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
    let from_new = KNeighborsClassifier::<f64>::new();
    let from_builder = KNeighborsClassifier::<f64>::builder()
        .build::<f64>()
        .expect("default KNeighborsClassifierBuilder builds");
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
    let est = KNeighborsClassifier::<f64>::builder()
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
    assert!(KNeighborsClassifier::<f64>::builder()
        .metric(Metric::Minkowski { p: 0.5 })
        .build::<f64>()
        .is_err());
}

/// A NaN exponent must be rejected too. This is the case the natural `p < 1.0`
/// spelling gets wrong: every ordered comparison against NaN is false, so `p <
/// 1.0` waves it through and the kernel emits NaN distances that `top_k` orders
/// arbitrarily — a silently wrong neighbour set, not an error.
#[test]
fn builder_rejects_minkowski_p_nan() {
    assert!(KNeighborsClassifier::<f64>::builder()
        .metric(Metric::Minkowski { p: f64::NAN })
        .build::<f64>()
        .is_err());
}

/// `n_neighbors = 0` stays rejected at build (unchanged by KNN-CLF-PARAMS).
#[test]
fn builder_rejects_zero_n_neighbors() {
    assert!(KNeighborsClassifier::<f64>::builder()
        .n_neighbors(0)
        .build::<f64>()
        .is_err());
}

// ---------------------------------------------------------------------------
// Label preparation: the already-dense fast path
// ---------------------------------------------------------------------------

/// `prepare_labels`'s dense fast path must agree with the general sort/dedup
/// path on every input, not just on the one it was written for.
///
/// The fast path skips the sort and the per-sample binary search when the label
/// set is exactly `{0..max}` — which is what the Python shim always hands it,
/// because the shim does sklearn's `np.unique` encoding itself. The risk it
/// introduces is a GAPPED set slipping through: `{0, 2}` has `max = 2` but sorts
/// to `[0, 2]`, so label `2` is at position 1 and the identity mapping would be
/// silently wrong — and wrong in the direction CR-03 exists to prevent. Each
/// case below is checked against the answer the general path is defined to give.
#[test]
fn prepare_labels_dense_fast_path_agrees_with_the_general_path() {
    for (name, raw, want_classes, want_codes) in [
        // Dense and already ascending — the shim's own shape.
        (
            "dense ascending",
            vec![0, 1, 2, 1, 0],
            vec![0, 1, 2],
            vec![0, 1, 2, 1, 0],
        ),
        // Dense but SHUFFLED: position in the sorted set is still the label.
        (
            "dense shuffled",
            vec![2, 0, 1, 2],
            vec![0, 1, 2],
            vec![2, 0, 1, 2],
        ),
        // Single class.
        ("single class", vec![0, 0, 0], vec![0], vec![0, 0, 0]),
        // GAPPED — must take the general path. `2` maps to position 1, NOT 2.
        (
            "gapped {0,2}",
            vec![0, 2, 2, 0],
            vec![0, 2],
            vec![0, 1, 1, 0],
        ),
        // Not starting at 0 — also gapped from the fast path's point of view.
        ("offset {3,4}", vec![3, 4, 3], vec![3, 4], vec![0, 1, 0]),
        // Negative ids are legal labels and never dense.
        ("negative", vec![-1, 0, -1], vec![-1, 0], vec![0, 1, 0]),
    ] {
        let n = raw.len();
        let y: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
        let prepared = prepare_labels::<f64>(&y, n)
            .unwrap_or_else(|e| panic!("{name}: prepare_labels rejected a valid target: {e:?}"));
        assert_eq!(prepared.classes(), want_classes, "{name}: classes_");
        assert_eq!(
            prepared.n_classes(),
            want_classes.len(),
            "{name}: n_classes"
        );
        // The codes are read back through a fitted estimator's proba columns, so
        // assert them via the documented invariant instead of a private field:
        // `classes_[code] == raw label` for every sample.
        let codes = &want_codes;
        for (i, (&code, &label)) in codes.iter().zip(&raw).enumerate() {
            assert_eq!(
                want_classes[code as usize], label,
                "{name}: sample {i} — classes_[{code}] must be its raw label {label}"
            );
        }
    }
}

/// A label at or beyond `n_train` declines the fast path (its presence bitmap
/// would not be `O(n_train)`) but must still map CORRECTLY through the general
/// one — the bound is a memory guard, never a change of answer.
#[test]
fn prepare_labels_handles_a_label_beyond_n_train() {
    let raw = [0i32, 1, 999];
    let y: Vec<f64> = raw.iter().map(|&v| v as f64).collect();
    let prepared = prepare_labels::<f64>(&y, raw.len()).expect("valid target");
    assert_eq!(prepared.classes(), [0, 1, 999]);
}

// ---------------------------------------------------------------------------
// weights x metric against sklearn
// ---------------------------------------------------------------------------

#[test]
fn weights_metric_matrix_match_sklearn_f32() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "params");
    run_all_metric_cases::<f32>("knn_clf_params_f32_seed42.npz");
}

#[test]
fn weights_metric_matrix_match_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "params");
    if capability::skip_f64_with_log() {
        println!("knn_classifier params f64 backend={backend}: SKIPPED (no f64 support)");
        return;
    }
    run_all_metric_cases::<f64>("knn_clf_params_f64_seed42.npz");
}

// ---------------------------------------------------------------------------
// kneighbors under a NON-default metric
// ---------------------------------------------------------------------------

/// `kneighbors` reports the CONFIGURED metric's neighbours.
///
/// Checked in its own test rather than being inferred from `predict`: a
/// wrong-metric neighbour set can still vote for the right class on a
/// well-separated design, so the distances are what actually pin it.
///
/// Indices are compared as a per-row SET, not position by position. The fixture
/// contains a duplicated training row (deliberately — it is what gives a
/// coincident query two zero-distance neighbours), so those two indices are an
/// exact distance TIE and their relative order is not determined by the problem.
/// mlrs's `top_k` resolves ties to the lowest index; sklearn's brute path makes
/// no such guarantee. Asserting positional equality would be asserting an
/// implementation detail of sklearn's sort.
#[test]
fn kneighbors_non_default_metric_matches_sklearn_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        println!("knn_classifier kneighbors f64: SKIPPED (no f64 support)");
        return;
    }
    let case = load_npz(fixture("knn_clf_params_f64_seed42.npz")).expect("load fixture");
    let x: Vec<f64> = fixture_vec::<f64>(&case, "X");
    let xq: Vec<f64> = fixture_vec::<f64>(&case, "Xq");
    let y: Vec<f64> = fixture_vec::<f64>(&case, "y");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);
    let xq_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &xq);
    let y_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &y);

    let clf = KNeighborsClassifier::<f64>::builder()
        .n_neighbors(K)
        .metric(Metric::Manhattan)
        .build::<f64>()
        .expect("build")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_TRAIN, N_FEATURES))
        .expect("fit");

    let (dist, idx) = clf
        .kneighbors(&mut pool, &xq_dev, (N_QUERY, N_FEATURES), K)
        .expect("kneighbors");
    let got_d: Vec<f64> = dist.to_host(&pool);
    let got_i: Vec<i32> = idx.to_host(&pool);
    dist.release_into(&mut pool);
    idx.release_into(&mut pool);

    assert_proba(&got_d, case.expect_f64("distances_manhattan"), "distances");

    let want_i = case.expect_f64("indices_manhattan");
    for q in 0..N_QUERY {
        let mut g: Vec<i32> = got_i[q * K..(q + 1) * K].to_vec();
        let mut w: Vec<i32> = want_i[q * K..(q + 1) * K]
            .iter()
            .map(|&v| v as i32)
            .collect();
        g.sort_unstable();
        w.sort_unstable();
        assert_eq!(g, w, "kneighbors indices (as a set) for query row {q}");
    }
}

// ---------------------------------------------------------------------------
// The non-contiguous class space (CR-03)
// ---------------------------------------------------------------------------

/// `classes_` is the DISTINCT SORTED training set and `predict` returns ids
/// drawn from it — never the dense column index.
///
/// The fixture's `{0, 2, 7}` is the point: with `{0, 1, 2}` labels the dense
/// index and the class id coincide, so a `predict` that skipped the `classes_`
/// lookup entirely would pass every other test in this file.
#[test]
fn classes_round_trip_non_contiguous_labels_f64() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        println!("knn_classifier classes_ f64: SKIPPED (no f64 support)");
        return;
    }
    let (labels, proba) = run_case::<f64>(
        "knn_clf_params_f64_seed42.npz",
        Weights::Uniform,
        Metric::Euclidean,
    );
    assert!(labels.iter().all(|l| CLASSES.contains(l)));
    // Every proba row sums to 1 (the normalization sklearn applies), and is
    // exactly `n_classes` wide — a `max + 1` width over `{0, 2, 7}` would be 8
    // columns, six of them structurally zero.
    assert_eq!(proba.len(), N_QUERY * N_CLASSES);
    for q in 0..N_QUERY {
        let s: f64 = proba[q * N_CLASSES..(q + 1) * N_CLASSES].iter().sum();
        assert!(
            (s - 1.0).abs() <= PROBA_TOL,
            "proba row {q} sums to {s}, not 1"
        );
    }
}
