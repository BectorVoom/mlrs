//! LINEAR-PERSIST (prototype) — safetensors save/load round-trips for the dense
//! linear estimators: `LinearRegression`, `Ridge`, `Lasso` and `ElasticNet`.
//!
//! Each exercises a different corner of the shared core. `LinearRegression` is
//! the minimal case — its whole fitted state IS the core, one target, no extra
//! scalars. `Ridge` adds the parts that are easy to get wrong: eight
//! hyperparameters including two `Option`s and two enums, three FITTED
//! diagnostics (`n_iter_`/`solver_`/`device_`), multi-target `coef_`, and the
//! one place the family genuinely disagrees — Ridge holds `coef_`
//! FEATURES-major while the file stores sklearn's TARGETS-major orientation.
//! `Lasso` and `ElasticNet` are the near-duplicate pair: two files that differ
//! by exactly one header key, which makes the `estimator` discriminator load
//! bearing rather than decorative.
//!
//! The gates, in the order they matter:
//!
//!   - `*_roundtrip_is_bit_exact` — `coef_`/`intercept_`/`fit_intercept` survive
//!     save→load with `==`, not a tolerance. Persistence has no numerical error
//!     budget: a round-trip that only matches to 1e-5 has a bug, and a band
//!     would hide it.
//!   - `roundtrip_preserves_predictions` — the reloaded model predicts
//!     identically, which is the property a user actually cares about.
//!   - `fit_intercept_false_roundtrips` — the one hyperparameter, whose value
//!     rides in `__metadata__` and is NOT recoverable from a zero intercept.
//!   - `f32_model_writes_a_half_size_file` — the dtype-tag claim, measured on
//!     real files rather than asserted in a comment.
//!   - `f32_file_loads_into_an_f64_model` — the other half of that claim: the
//!     tag makes the file self-describing, so the width is a load-time choice.
//!   - `the_file_is_the_model_and_little_else` — the minimal-file-size claim,
//!     measured as payload vs total: nothing derivable is stored, and the only
//!     non-payload bytes are the safetensors header itself.
//!   - `the_load_path_is_zero_copy` — the `AlignedBytes` claim, which nothing in
//!     a round-trip assertion would reveal.
//!   - `saving_twice_produces_an_identical_model` — byte-level determinism, and
//!     the gate on the `third_party/safetensors` `BTreeMap` patch.
//!   - the rejection gates — a Naive Bayes file (the OTHER container), a sibling
//!     linear estimator's file, a header whose extents disagree, and a
//!     zero-extent header. The file is untrusted input (T-04-01-01), so an
//!     inconsistent header must be a typed error, never an out-of-bounds read at
//!     predict time.
//!   - `ridge_every_persisted_field_roundtrips` — save→load→save is byte-stable,
//!     which covers the four hyperparameters (`copy_X`, `tol`, `max_iter`,
//!     `random_state`) that have no public accessor to compare directly.
//!   - `ridge_coef_is_stored_in_sklearn_orientation` — the layout gate: the file
//!     holds `[n_targets, n_features]` even though Ridge holds the transpose.
//!   - `lasso_and_elastic_net_files_do_not_cross_load` — the sharpest
//!     discriminator case in the family: `ElasticNet(l1_ratio=1)` IS `Lasso`, so
//!     the two files differ by ONE header key and nothing else.
//!   - `an_elastic_net_file_without_l1_ratio_is_rejected` — that key is
//!     REQUIRED, never defaulted: silently substituting `0.5` would hand back a
//!     model with a different penalty than the one that was saved.
//!   - `save_leaves_no_temporary_behind` — the write-then-rename path.
//!
//! Fixtures are generated in-test rather than loaded from an oracle `.npz`:
//! these gates are about the CONTAINER, and comparing a model against itself
//! needs no sklearn reference. The sklearn-parity gates for the fit itself live
//! in `linear_regression_test.rs`.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::elastic_net::ElasticNet;
use mlrs_algos::linear::lasso::Lasso;
use mlrs_algos::linear::linear_persist::{
    AlignedBytes, LinearFile, LinearWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::linear::ridge::{Ridge, RidgeSolver};
use mlrs_algos::naive_bayes::GaussianNB;
use mlrs_algos::typestate::{Fit, Fitted, Predict};
use mlrs_backend::capability;
use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The fixture geometry — `n_samples.max(n_features) <= 256`, so the fit takes
/// `fit_direct_svd`. Which solver ran is irrelevant to the container, but a
/// deterministic 12×4 problem keeps the file small enough to reason about byte
/// by byte in the size gates below.
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;

/// A deterministic, well-conditioned regression fixture.
///
/// Hand-written rather than seeded-random on purpose: a persistence round-trip
/// is exact or broken, and an RNG would only add a way for the two arms to
/// disagree for reasons that have nothing to do with the file. The columns are
/// mutually non-collinear so the SVD pseudo-inverse returns the ordinary
/// full-rank solution rather than a minimum-norm one — again not the container's
/// business, but it keeps the fitted values ordinary.
fn fixture<F: Pod>() -> (Vec<F>, Vec<F>) {
    let rows: [[f64; N_FEATURES]; N_SAMPLES] = [
        [0.31, -1.24, 0.88, 2.10],
        [-0.75, 0.42, 1.63, -0.19],
        [1.28, 0.07, -0.54, 0.93],
        [0.02, 1.85, 0.31, -1.47],
        [-1.11, -0.68, 2.05, 0.24],
        [0.96, 1.32, -1.18, 0.57],
        [2.14, -0.29, 0.46, -0.82],
        [-0.38, 0.71, 1.09, 1.66],
        [1.53, -1.90, 0.12, 0.35],
        [-0.64, 0.15, -1.37, 1.28],
        [0.87, 2.03, 0.79, -0.41],
        [-1.42, 0.56, 1.94, 0.68],
    ];
    let y: Vec<F> = rows
        .iter()
        .map(|r| {
            mlrs_core::f64_to_host::<F>(3.0 + 1.5 * r[0] - 2.0 * r[1] + 0.75 * r[2] + 0.5 * r[3])
        })
        .collect();
    let x = rows
        .iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    (x, y)
}

fn fit_ols<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    fit_intercept: bool,
) -> LinearRegression<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    LinearRegression::<F>::builder()
        .fit_intercept(fit_intercept)
        .build::<F>()
        .expect("LinearRegression builds with valid hyperparameters")
        .fit(pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("LinearRegression fits the fixture")
}

/// Device `predict` over the fixture's own rows.
fn predictions<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &LinearRegression<F, Fitted>,
) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let (x, _) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    model
        .predict(pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds on the training geometry")
        .to_host(pool)
}

/// A default-hyperparameter `Ridge` over the same fixture.
fn fit_ridge<F>(pool: &mut BufferPool<ActiveRuntime>) -> Ridge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    Ridge::<F>::builder()
        .build::<F>()
        .expect("Ridge builds with default hyperparameters")
        .fit(pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge fits the fixture")
}

/// A `Ridge` with EVERY hyperparameter moved off its default, so a round-trip
/// that drops one is visible.
///
/// `svd` rather than an iterative solver on purpose: `max_iter`/`random_state`
/// are inert for it, which is exactly what makes them worth pinning — a field
/// the fit never reads is the one a `save` is most likely to forget, and the
/// only way to catch that is to store it and compare.
fn fit_ridge_non_default<F>(pool: &mut BufferPool<ActiveRuntime>) -> Ridge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    Ridge::<F>::builder()
        .alpha(2.5)
        .fit_intercept(false)
        .copy_x(false)
        .max_iter(Some(77))
        .tol(1e-6)
        .solver(RidgeSolver::Svd)
        .positive(false)
        .random_state(Some(12_345))
        .device(Device::Gpu)
        .build::<F>()
        .expect("Ridge builds with these hyperparameters")
        .fit(pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Ridge fits the fixture")
}

/// The fixture's `y`, widened to `n_targets` columns by scaling each one — a
/// genuine multi-target problem whose columns have DIFFERENT solutions, so a
/// transposed or truncated `coef_` cannot pass by coincidence.
fn multi_target_y<F: Pod>(n_targets: usize) -> Vec<F> {
    let (_, y) = fixture::<f64>();
    let mut out = Vec::with_capacity(N_SAMPLES * n_targets);
    for &v in &y {
        for t in 0..n_targets {
            out.push(mlrs_core::f64_to_host::<F>(v * (1.0 + t as f64)));
        }
    }
    out
}

fn fit_ridge_multi<F>(pool: &mut BufferPool<ActiveRuntime>, n_targets: usize) -> Ridge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, _) = fixture::<F>();
    let y = multi_target_y::<F>(n_targets);
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    Ridge::<F>::builder()
        .build::<F>()
        .expect("Ridge builds with default hyperparameters")
        .fit_multi_target_with_sample_weight(
            pool,
            &x_dev,
            &y_dev,
            (N_SAMPLES, N_FEATURES),
            n_targets,
            None,
        )
        .expect("Ridge fits the multi-target fixture")
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_is_bit_exact_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_ols::<f64>(&mut pool, true);
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: LinearRegression<f64, Fitted> =
        LinearRegression::load(&mut pool, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.coef(&pool),
        fitted.coef(&pool),
        "coef_ must round-trip exactly"
    );
    assert_eq!(
        loaded.intercept(&pool),
        fitted.intercept(&pool),
        "intercept_ must round-trip exactly"
    );
}

#[test]
fn roundtrip_is_bit_exact_f32() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_ols::<f32>(&mut pool, true);
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: LinearRegression<f32, Fitted> =
        LinearRegression::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        loaded.coef(&pool),
        fitted.coef(&pool),
        "coef_ must round-trip exactly at f32 too — the file stores F32, not a widened copy"
    );
    assert_eq!(
        loaded.intercept(&pool),
        fitted.intercept(&pool),
        "intercept_ must round-trip exactly"
    );
}

#[test]
fn roundtrip_preserves_predictions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_ols::<f32>(&mut pool, true);
    let want = predictions(&mut pool, &fitted);
    fitted.save(&pool, &path).expect("save succeeds");

    // The property a user actually cares about: identical coefficients through
    // an identical `predict` must give identical outputs, bit for bit.
    let loaded: LinearRegression<f32, Fitted> =
        LinearRegression::load(&mut pool, &path).expect("load succeeds");
    assert_eq!(
        predictions(&mut pool, &loaded),
        want,
        "a reloaded model must predict identically"
    );
}

#[test]
fn fit_intercept_false_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no_intercept.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // `fit_intercept = false` is the one hyperparameter in the file, and it is
    // NOT recoverable from the fitted state: a model that simply happened to fit
    // a zero intercept is a different model. It rides in `__metadata__`, so this
    // is the gate that the scalar half of the container round-trips at all.
    let fitted = fit_ols::<f32>(&mut pool, false);
    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: LinearRegression<f32, Fitted> =
        LinearRegression::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        loaded.intercept(&pool),
        0.0f32,
        "an unfitted intercept is stored as zero, not omitted"
    );
    assert_eq!(
        predictions(&mut pool, &loaded),
        predictions(&mut pool, &fitted),
        "the reloaded no-intercept model must predict identically"
    );

    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = LinearFile::parse(&raw, "linear_regression").expect("parse succeeds");
    assert_eq!(
        file.metadata()
            .get("param:fit_intercept")
            .map(String::as_str),
        Some("false"),
        "fit_intercept must be recorded in __metadata__, not inferred at load"
    );
}

// ---------------------------------------------------------------------------
// Ridge — the same core, plus eight hyperparameters and a transposed layout
// ---------------------------------------------------------------------------

#[test]
fn ridge_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ridge.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_ridge::<f32>(&mut pool);
    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: Ridge<f32, Fitted> = Ridge::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        loaded.coef(&pool),
        fitted.coef(&pool),
        "coef_ must round-trip exactly"
    );
    assert_eq!(
        loaded.intercept(&pool),
        fitted.intercept(&pool),
        "intercept_ must round-trip exactly"
    );

    // The FITTED diagnostics, not the inputs: `param:solver` is `auto` and
    // `param:device` is `Auto` on a default Ridge, so neither would tell a
    // reloaded model's user what actually produced the numbers. These must
    // survive independently of the knobs that led to them.
    assert_eq!(
        loaded.solver(),
        fitted.solver(),
        "solver_ must round-trip — 'auto' already resolved"
    );
    assert_eq!(
        loaded.device(),
        fitted.device(),
        "device_ must round-trip — the arm that actually ran"
    );
    assert_eq!(
        loaded.n_iter(),
        fitted.n_iter(),
        "n_iter_ must round-trip, including when it is None"
    );
}

#[test]
fn ridge_roundtrip_preserves_predictions() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ridge.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (x, _) = fixture::<f32>();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);

    let fitted = fit_ridge::<f32>(&mut pool);
    let want = fitted
        .predict(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds")
        .to_host(&pool);
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: Ridge<f32, Fitted> = Ridge::load(&mut pool, &path).expect("load succeeds");
    let got = loaded
        .predict(&mut pool, &x_dev, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds")
        .to_host(&pool);
    assert_eq!(got, want, "a reloaded Ridge must predict identically");
}

#[test]
fn ridge_every_persisted_field_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Four of Ridge's hyperparameters — `copy_X`, `tol`, `max_iter`,
    // `random_state` — have no public accessor on the fitted estimator, so they
    // cannot be compared field by field. Save→load→save closes that hole
    // wholesale: the file is a deterministic function of the model (see
    // `saving_twice_produces_an_identical_model`), so if re-saving the RELOADED
    // model reproduces the original bytes, every persisted field survived. A
    // dropped `max_iter` or a defaulted `copy_X` changes the header and fails
    // here.
    let fitted = fit_ridge_non_default::<f32>(&mut pool);
    fitted.save(&pool, &first).expect("save succeeds");
    let loaded: Ridge<f32, Fitted> = Ridge::load(&mut pool, &first).expect("load succeeds");
    loaded.save(&pool, &second).expect("re-save succeeds");

    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "save -> load -> save must be byte-stable: any field that failed to \
         round-trip changes the header"
    );

    // And spot-check the ones that DO have accessors, so a failure above points
    // somewhere rather than just saying "the bytes differ".
    assert_eq!(loaded.solver(), RidgeSolver::Svd, "solver_ must round-trip");
    assert_eq!(loaded.device(), fitted.device(), "device_ must round-trip");
}

#[test]
fn ridge_multi_target_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ridge_multi.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Three targets, so the features-major ↔ targets-major hop is a real
    // permutation rather than the identity it is at `n_targets == 1`. The
    // targets have different solutions, so a transposed round-trip cannot pass
    // by coincidence.
    let fitted = fit_ridge_multi::<f32>(&mut pool, 3);
    assert_eq!(
        fitted.n_targets(),
        3,
        "the fixture is genuinely multi-target"
    );
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: Ridge<f32, Fitted> = Ridge::load(&mut pool, &path).expect("load succeeds");
    assert_eq!(loaded.n_targets(), 3, "n_targets must round-trip");
    assert_eq!(
        loaded.coef_multi(&pool),
        fitted.coef_multi(&pool),
        "the n_features x n_targets coef_ must round-trip exactly, in Ridge's own layout"
    );
    assert_eq!(
        loaded.intercept_multi(&pool),
        fitted.intercept_multi(&pool),
        "the per-target intercept_ must round-trip exactly"
    );
}

#[test]
fn ridge_coef_is_stored_in_sklearn_orientation() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ridge_multi.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n_targets = 3;
    let fitted = fit_ridge_multi::<f32>(&mut pool, n_targets);
    let in_memory = fitted.coef_multi(&pool); // n_features x n_targets
    fitted.save(&pool, &path).expect("save succeeds");

    // The layout claim, checked against the FILE rather than against a
    // round-trip — a save and a load that are transposed the same way agree
    // with each other and are both wrong. `coef_` on disk must be sklearn's
    // `(n_targets, n_features)`, so that `safetensors.numpy.load_file(path)`
    // in Python hands back the array `Ridge.coef_` would be.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = LinearFile::parse(&raw, "ridge").expect("parse succeeds");
    let coef = file.tensor("coef_").expect("coef_ is present");
    assert_eq!(
        coef.shape(),
        &[n_targets, N_FEATURES],
        "the file stores coef_ as [n_targets, n_features], sklearn's orientation"
    );

    let on_disk: &[f32] = bytemuck::cast_slice(coef.data());
    for t in 0..n_targets {
        for f in 0..N_FEATURES {
            assert_eq!(
                on_disk[t * N_FEATURES + f],
                in_memory[f * n_targets + t],
                "coef_[{t}][{f}] on disk must be the transpose of the in-memory buffer"
            );
        }
    }
}

#[test]
fn a_linear_regression_file_does_not_load_as_ridge() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_ols::<f32>(&mut pool, true)
        .save(&pool, &path)
        .expect("save succeeds");

    // The reason the `estimator` discriminator has to exist at all: these two
    // are the SAME container with the SAME two tensors, and a
    // `LinearRegression` file parses perfectly as the shared core. Loading it
    // as a Ridge would silently invent every hyperparameter — except that the
    // discriminator is checked before any tensor is fetched.
    let err = match Ridge::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a LinearRegression file must not load as a Ridge"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "ridge" && found == "linear_regression"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Lasso / ElasticNet — two files that differ by exactly one header key
// ---------------------------------------------------------------------------

/// A `Lasso` whose penalty is small enough to leave a non-degenerate solution.
///
/// `alpha` matters here in a way it does not for Ridge: at sklearn's default
/// `alpha = 1.0` this fixture's L1 penalty drives every coefficient to zero, and
/// a round-trip of an all-zero `coef_` would pass no matter how broken the
/// container was. The tests assert non-degeneracy explicitly rather than trust
/// this number.
fn fit_lasso<F>(pool: &mut BufferPool<ActiveRuntime>, alpha: f64) -> Lasso<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    Lasso::<F>::builder()
        .alpha(alpha)
        .build::<F>()
        .expect("Lasso builds with valid hyperparameters")
        .fit(pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Lasso fits the fixture")
}

fn fit_elastic_net<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    alpha: f64,
    l1_ratio: f64,
) -> ElasticNet<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y) = fixture::<F>();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &y);
    ElasticNet::<F>::builder()
        .alpha(alpha)
        .l1_ratio(l1_ratio)
        .build::<F>()
        .expect("ElasticNet builds with valid hyperparameters")
        .fit(pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("ElasticNet fits the fixture")
}

/// A fitted `coef_` that is all zeros makes a round-trip assertion vacuous —
/// every container, however broken, reproduces zeros. Both L1 estimators can
/// land there for a large enough `alpha`, so every test below states the
/// precondition rather than assuming it.
fn assert_not_degenerate(coef: &[f32], what: &str) {
    assert!(
        coef.iter().any(|&c| c != 0.0),
        "{what} shrank every coefficient to zero — the round-trip assertions \
         would pass vacuously; lower alpha"
    );
}

#[test]
fn lasso_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("lasso.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_lasso::<f32>(&mut pool, 0.05);
    assert_not_degenerate(&fitted.coef(&pool), "Lasso");
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: Lasso<f32, Fitted> = Lasso::load(&mut pool, &path).expect("load succeeds");
    assert_eq!(
        loaded.coef(&pool),
        fitted.coef(&pool),
        "coef_ must round-trip exactly"
    );
    assert_eq!(
        loaded.intercept(&pool),
        fitted.intercept(&pool),
        "intercept_ must round-trip exactly"
    );
}

#[test]
fn elastic_net_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("enet.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_elastic_net::<f32>(&mut pool, 0.05, 0.25);
    assert_not_degenerate(&fitted.coef(&pool), "ElasticNet");
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: ElasticNet<f32, Fitted> =
        ElasticNet::load(&mut pool, &path).expect("load succeeds");
    assert_eq!(
        loaded.coef(&pool),
        fitted.coef(&pool),
        "coef_ must round-trip exactly"
    );
    assert_eq!(
        loaded.intercept(&pool),
        fitted.intercept(&pool),
        "intercept_ must round-trip exactly"
    );
}

#[test]
fn cd_pair_every_persisted_field_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Neither estimator exposes `alpha`, `max_iter`, `tol` or `l1_ratio` on the
    // fitted value, so save→load→save byte-stability is the only way to gate
    // them — the same argument `ridge_every_persisted_field_roundtrips` makes.
    // Every hyperparameter is off its default here, so a `load` that forgot one
    // and fell back to the default changes the header and fails.
    let lasso_a = dir.path().join("lasso_a.safetensors");
    let lasso_b = dir.path().join("lasso_b.safetensors");
    let (x, y) = fixture::<f32>();
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let y_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &y);

    Lasso::<f32>::builder()
        .alpha(0.05)
        .fit_intercept(false)
        .max_iter(250)
        .tol(1e-6)
        .build::<f32>()
        .expect("Lasso builds")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("Lasso fits")
        .save(&pool, &lasso_a)
        .expect("save succeeds");
    let reloaded: Lasso<f32, Fitted> = Lasso::load(&mut pool, &lasso_a).expect("load succeeds");
    reloaded.save(&pool, &lasso_b).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&lasso_a).expect("read"),
        std::fs::read(&lasso_b).expect("read"),
        "Lasso save -> load -> save must be byte-stable"
    );

    let enet_a = dir.path().join("enet_a.safetensors");
    let enet_b = dir.path().join("enet_b.safetensors");
    ElasticNet::<f32>::builder()
        .alpha(0.05)
        .l1_ratio(0.25)
        .fit_intercept(false)
        .max_iter(250)
        .tol(1e-6)
        .build::<f32>()
        .expect("ElasticNet builds")
        .fit(&mut pool, &x_dev, Some(&y_dev), (N_SAMPLES, N_FEATURES))
        .expect("ElasticNet fits")
        .save(&pool, &enet_a)
        .expect("save succeeds");
    let reloaded: ElasticNet<f32, Fitted> =
        ElasticNet::load(&mut pool, &enet_a).expect("load succeeds");
    reloaded.save(&pool, &enet_b).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&enet_a).expect("read"),
        std::fs::read(&enet_b).expect("read"),
        "ElasticNet save -> load -> save must be byte-stable, l1_ratio included"
    );
}

#[test]
fn lasso_and_elastic_net_files_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let lasso_path = dir.path().join("lasso.safetensors");
    let enet_path = dir.path().join("enet.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // The sharpest case in the family. `ElasticNet(l1_ratio=1)` IS `Lasso` —
    // same objective, same solver, same fitted state — so these two files differ
    // by ONE header key and nothing else. A `Lasso` file loaded as an ElasticNet
    // would be missing only `param:l1_ratio`; an ElasticNet file loaded as a
    // Lasso would parse completely and silently discard the L2 half of the
    // penalty it was fitted with. Only the `estimator` discriminator separates
    // them, and it is checked before any tensor is fetched.
    fit_lasso::<f32>(&mut pool, 0.05)
        .save(&pool, &lasso_path)
        .expect("save succeeds");
    fit_elastic_net::<f32>(&mut pool, 0.05, 1.0)
        .save(&pool, &enet_path)
        .expect("save succeeds");

    let err = match ElasticNet::<f32, Fitted>::load(&mut pool, &lasso_path) {
        Ok(_) => panic!("a Lasso file must not load as an ElasticNet"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "elastic_net" && found == "lasso"
        ),
        "expected WrongEstimator, got {err:?}"
    );

    let err = match Lasso::<f32, Fitted>::load(&mut pool, &enet_path) {
        Ok(_) => panic!("an ElasticNet file must not load as a Lasso, even at l1_ratio = 1"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "lasso" && found == "elastic_net"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn an_elastic_net_file_without_l1_ratio_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no_l1_ratio.safetensors");

    // Everything an ElasticNet file needs EXCEPT the one key that distinguishes
    // it from a Lasso. Defaulting it to 0.5 would hand back a model with a
    // different penalty than the one that was saved and no way to tell, so the
    // key is required and its absence is a typed error naming it.
    let coef = vec![0.5f32; N_FEATURES];
    let intercept = vec![1.25f32];
    let mut w = LinearWriter::new("elastic_net");
    w.scalar_bool("param:fit_intercept", true);
    w.scalar_f64("param:alpha", 0.05);
    w.scalar_usize("param:max_iter", 1000);
    w.scalar_f64("param:tol", 1e-4);
    w.tensor(
        "coef_",
        TensorRef::floats(&coef, vec![1, N_FEATURES]).unwrap(),
    );
    w.tensor(
        "intercept_",
        TensorRef::floats(&intercept, vec![1]).unwrap(),
    );
    w.write(&path).expect("write succeeds");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match ElasticNet::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("an ElasticNet file with no l1_ratio must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:l1_ratio"),
        "expected BadMetadata naming param:l1_ratio, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The size and speed claims
// ---------------------------------------------------------------------------

#[test]
fn f32_model_writes_a_half_size_file() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let small = dir.path().join("f32.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    fit_ols::<f32>(&mut pool, true)
        .save(&pool, &small)
        .expect("save succeeds");
    let f32_len = std::fs::metadata(&small).unwrap().len();

    if capability::skip_f64_with_log() {
        return;
    }
    let big = dir.path().join("f64.safetensors");
    fit_ols::<f64>(&mut pool, true)
        .save(&pool, &big)
        .expect("save succeeds");
    let f64_len = std::fs::metadata(&big).unwrap().len();

    // `coef_` (4 values) and `intercept_` (1) are the whole payload, so the f32
    // file must be exactly 5·(8−4) = 20 bytes smaller. An exact difference
    // rather than a ratio: the two headers are the same length (`"F32"` and
    // `"F64"` are both three characters, and the offsets happen to have equal
    // digit counts at this geometry), so this pins the claim that ONLY the
    // model's own float width changed.
    assert_eq!(
        f64_len - f32_len,
        20,
        "an f32 model must store F32 tensors: f32 file {f32_len} B, f64 file {f64_len} B"
    );
}

#[test]
fn f32_file_loads_into_an_f64_model() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("f32.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_ols::<f32>(&mut pool, true);
    let want_coef: Vec<f64> = fitted.coef(&pool).iter().map(|&v| v as f64).collect();
    let want_intercept = fitted.intercept(&pool) as f64;
    fitted.save(&pool, &path).expect("save succeeds");

    // The dtype tag makes the file self-describing, so the element width is a
    // LOAD-time choice: train on a GPU in f32, evaluate in f64. The values are
    // the f32 ones exactly — widening f32→f64 is lossless.
    let widened: LinearRegression<f64, Fitted> =
        LinearRegression::load(&mut pool, &path).expect("an f32 file loads into an f64 model");
    assert_eq!(
        widened.coef(&pool),
        want_coef,
        "widening an f32 payload to f64 must be exact"
    );
    assert_eq!(
        widened.intercept(&pool),
        want_intercept,
        "widening the intercept must be exact too"
    );
}

#[test]
fn the_file_is_the_model_and_little_else() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_ols::<f32>(&mut pool, true)
        .save(&pool, &path)
        .expect("save succeeds");

    // The minimal-file-size claim, stated as an arithmetic identity rather than
    // a vibe: the payload is EXACTLY `(n_features + 1)` f32 values — no padding,
    // no widening, and no second copy of `n_features`, which is read back off
    // `coef_`'s shape. Everything else in the file is the safetensors header,
    // which for a two-tensor model is a couple of hundred bytes of JSON and does
    // not grow with the model.
    let payload = (N_FEATURES + 1) * size_of::<f32>();
    let total = std::fs::metadata(&path).unwrap().len() as usize;
    let header = total - payload;
    assert!(
        header < 320,
        "the non-payload overhead must be the header alone, got {header} B \
         over a {payload} B payload ({total} B total)"
    );

    // And the header's own accounting agrees: the two tensors cover the whole
    // data section back to back.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = LinearFile::parse(&raw, "linear_regression").expect("parse succeeds");
    let coef = file.tensor("coef_").expect("coef_ is present");
    let intercept = file.tensor("intercept_").expect("intercept_ is present");
    assert_eq!(
        coef.shape(),
        &[1, N_FEATURES],
        "coef_ is one fused [n_targets, n_features] block"
    );
    assert_eq!(intercept.shape(), &[1], "intercept_ is [n_targets]");
    assert_eq!(
        coef.data().len() + intercept.data().len(),
        payload,
        "the two tensors must account for the entire payload, with no padding"
    );
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_ols::<f64>(&mut pool, true)
        .save(&pool, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: every 8-byte tensor in a
    // file this crate wrote can be reinterpreted from the file buffer with NO
    // copy, so `load` hands the file's own bytes to `DeviceArray::from_host`.
    // safetensors pads its header to a multiple of 8 and emits tensors in
    // descending dtype width, so an 8-aligned base is all it takes — but a
    // `Vec<u8>` from `fs::read` is only guaranteed 1-aligned, which would push
    // every tensor onto the copying fallback in `cast_bytes`. Nothing about that
    // is visible in a round-trip assertion, so it is gated here directly.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = LinearFile::parse(&raw, "linear_regression").expect("parse succeeds");
    for name in ["coef_", "intercept_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert!(
            bytemuck::try_cast_slice::<u8, f64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[f64] without a copy"
        );
    }
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let fitted = fit_ols::<f32>(&mut pool, true);
    fitted.save(&pool, &first).expect("save succeeds");
    fitted.save(&pool, &second).expect("save succeeds");

    // RAW BYTES, not just contents: a model file must be a deterministic
    // function of the model, so it can be content-addressed and deduplicated.
    //
    // This is the gate on the `third_party/safetensors` patch. Stock
    // safetensors serializes `__metadata__` out of a std `HashMap` whose
    // iteration order is randomly seeded, which makes two saves of one model
    // differ in header key order — semantically identical, byte-wise not. The
    // vendored fork retypes those maps to `BTreeMap`. If the `[patch.crates-io]`
    // entry is ever dropped, this assertion is what fails.
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

// ---------------------------------------------------------------------------
// Rejection gates — the file is untrusted input
// ---------------------------------------------------------------------------

#[test]
fn a_naive_bayes_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // The cross-FAMILY gate. Both containers are safetensors files written by
    // this crate with the same writer; only the `format` discriminator
    // (`mlrs-nb` vs `mlrs-linear`) separates them, and it is checked before any
    // tensor is fetched — so this reports what the file actually is rather than
    // a missing-`coef_` error that reads like corruption.
    let (x, _) = fixture::<f32>();
    let labels: Vec<f32> = (0..N_SAMPLES).map(|i| (i % 3) as f32).collect();
    GaussianNB::<f32>::builder()
        .build::<f32>()
        .expect("GaussianNB builds")
        .fit_from_host_slice(&mut pool, &x, &labels, (N_SAMPLES, N_FEATURES), None)
        .expect("GaussianNB fits the fixture")
        .save(&pool, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("an mlrs-nb file must not load as a linear model"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-linear"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn a_sibling_estimators_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ridge.safetensors");

    // A real file from a DIFFERENT member of the family. The two carry the same
    // tensors, so `read_linear_core` would accept this one outright — without
    // the `estimator` discriminator it would load, silently discarding
    // everything Ridge stores beyond the shared core.
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_ridge::<f32>(&mut pool)
        .save(&pool, &path)
        .expect("save succeeds");

    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a Ridge file must not load as a LinearRegression"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "linear_regression" && found == "ridge"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn an_inconsistent_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("tampered.safetensors");

    // `coef_` declares one target while `intercept_` holds three. `coef_`
    // defines the geometry and every other extent is measured against it, so
    // this is a typed error — NOT an out-of-bounds read the first time the model
    // is used (T-04-01-01).
    let coef = vec![0.5f32; N_FEATURES];
    let intercept = vec![1.0f32, 2.0, 3.0];
    let mut w = LinearWriter::new("linear_regression");
    w.scalar_bool("param:fit_intercept", true);
    w.tensor(
        "coef_",
        TensorRef::floats(&coef, vec![1, N_FEATURES]).unwrap(),
    );
    w.tensor(
        "intercept_",
        TensorRef::floats(&intercept, vec![3]).unwrap(),
    );
    w.write(&path).expect("write succeeds");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a header whose extents disagree must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_multi_target_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("multi.safetensors");

    // The shared core admits `n_targets > 1` for the multi-output members of the
    // family. OLS is single-target, so it must refuse rather than load a matrix
    // whose extra rows `predict` would never read.
    let coef = vec![0.5f32; 3 * N_FEATURES];
    let intercept = vec![1.0f32, 2.0, 3.0];
    let mut w = LinearWriter::new("linear_regression");
    w.scalar_bool("param:fit_intercept", true);
    w.tensor(
        "coef_",
        TensorRef::floats(&coef, vec![3, N_FEATURES]).unwrap(),
    );
    w.tensor(
        "intercept_",
        TensorRef::floats(&intercept, vec![3]).unwrap(),
    );
    w.write(&path).expect("write succeeds");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a 3-target file must not load as single-target OLS"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_zero_extent_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("empty.safetensors");

    // A `[1, 0]` model cannot predict anything, and an empty upload is a
    // landmine on the device backends — so the extent check runs before any
    // value reaches `DeviceArray::from_host`.
    let empty: Vec<f32> = Vec::new();
    let intercept = vec![0.0f32];
    let mut w = LinearWriter::new("linear_regression");
    w.scalar_bool("param:fit_intercept", true);
    w.tensor("coef_", TensorRef::floats(&empty, vec![1, 0]).unwrap());
    w.tensor(
        "intercept_",
        TensorRef::floats(&intercept, vec![1]).unwrap(),
    );
    w.write(&path).expect("write succeeds");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a zero-feature model must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_missing_file_reports_its_path() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("absent.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match LinearRegression::<f32, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("loading a nonexistent file must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::Io { path: p, .. } if p == &path),
        "an I/O error must name the file it happened on, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_ols::<f32>(&mut pool, true)
        .save(&pool, &path)
        .expect("save succeeds");

    // `save` writes to a sibling temporary and renames it into place so an
    // interrupted write cannot replace a good model with a truncated one; the
    // temporary must not survive a successful save.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains("mlrs-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a successful save must leave no temporary file, found {leftovers:?}"
    );
    assert!(Path::new(&path).exists(), "the model file must exist");
}
